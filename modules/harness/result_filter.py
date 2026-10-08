"""
工具结果过滤
============
工具返回什么，不直接等于模型看到什么。

【为什么要过滤】
    1. 防上下文膨胀：一个 data_query 可能返回几千字，全部回灌进下一轮 prompt，
       几轮就把上下文窗口吃满（这就是"上下文膨胀"的主因之一）。
    2. 防越权泄漏：工具按自己的权限取回了完整记录（含手机号、薪资），
       但当前身份的 mask_level 是 2，模型不该看到明文。
    3. 防噪声干扰：超长结果里真正有用的往往只有前几行。

【过滤三步】
    脱敏（按 principal.mask_level） → 裁剪（按 max_result_chars） → 标注（meta 里写清楚被改过）
"""
from __future__ import annotations           # 允许类型注解里用未导入的名字（Python 3.7+ 兼容写法）

from dataclasses import dataclass            # dataclass 装饰器：ResultFilterConfig 用它
from typing import Any, Optional             # 标准库类型：Any 任意 / Optional 可空

from modules.common.logger import get_logger # 统一日志工厂
from modules.tools.base_tool import ToolResult  # 工具执行结果统一结构

_logger = get_logger("harness.result_filter")  # 模块级 logger，名字 = "harness.result_filter"

# 裁剪标记：模型看到它就知道"后面还有，但没给我"，不会误以为结果只有这么点
_TRUNCATED_HINT = "…(结果已裁剪，共 {total} 字符，已保留前 {kept} 字符)"  # 裁剪后缀模板：total=原长，kept=保留长度


# 【什么】结果过滤器配置
# 【作用】把裁剪上限/脱敏开关/敏感词表集中成一个可注入的配置对象
# 【为什么】配置独立成 dataclass 后，Harness 装配时从 settings 一次性映射，过滤器本体不读全局配置
@dataclass
class ResultFilterConfig:
    max_result_chars: int = 2000        # 成功结果最大字符数
    max_error_chars: int = 500          # 错误信息最大字符数
    mask_enabled: bool = True           # 是否按 principal.mask_level 脱敏
    sensitive_words: tuple = ()         # 额外要打码的词


# 【什么】结果过滤器
# 【作用】把工具返回值变成"可以安全回灌上下文"的形态
# 【为什么】脱敏保安全、裁剪保上下文窗口、标注保可观测，三件事必须集中在一处统一做
class ResultFilter:
    """工具结果裁剪 + 脱敏。

    【输入输出对照】
        f = ResultFilter(max_result_chars=20)
        f.filter(ToolResult(True, result="abcdefghijklmnopqrstuvwxyz"))
        → ToolResult(True, result="abcdefghijklmnopqrst…(结果已裁剪，共 26 字符，已保留前 20 字符)")

        f.filter(ToolResult(True, result={"phone": "13812345678"}), principal=<mask_level=2>)
        → result 里 phone 被打码（复用 modules.enterprise.masking.mask_record）

    【处理顺序】脱敏 → 转字符串 → 裁剪 → meta 标注。
        顺序不能换：必须先脱敏再转字符串，否则明文手机号会落进文本；
        裁剪放最后，保证被切掉的只是"多余长度"而不是敏感字段。
    """

    # 【什么】构造过滤器
    # 【作用】注入配置；不传 config 时使用默认值（成功结果 2000 字 / 错误 500 字 / 脱敏开启）
    def __init__(self, config: Optional[ResultFilterConfig] = None):
        self.config = config or ResultFilterConfig()     # 没传配置就用默认值

    # 【什么】过滤单个结果
    # 【作用】失败结果只裁错误信息；成功结果先脱敏再裁剪
    # 【参数】result    —— 工具执行结果（含 success / result / error / meta）
    #        principal —— 当前调用身份，只用它的 mask_level；传 None 表示无身份，不脱敏
    #        max_chars —— 本次调用临时指定的字符上限，优先级高于配置；0 表示回退到配置值
    # 【返回】原地修改后的同一个 result 对象，改动痕迹都写在它的 meta 里
    def filter(self, result: ToolResult, principal: Any = None,
               max_chars: int = 0) -> ToolResult:
        # 空结果直接放行：没有可过滤的内容，也避免后面访问属性时报错
        if result is None:                               # 结果对象为 None
            return result                                # 原样返回
        # 本次生效的上限：调用参数 > 配置值；一路 or 下来为 0 时表示"不裁剪"
        limit = int(max_chars or self.config.max_result_chars or 0)  # 生效裁剪上限；0 = 不裁剪
        # 拷贝一份 meta 再写，避免多个结果共用同一个字典时互相污染
        meta = dict(getattr(result, "meta", {}) or {})   # meta 副本；None 兜底成空 dict

        # ---- 分支一：失败结果 ------------------------------------------
        # 失败时没有有效 result，只需防止超长报错（如整段堆栈）灌爆上下文
        if not result.success:                           # 失败分支
            error = str(result.error or "")              # 错误信息转字符串；None 兜底空串
            # limit 为 0（关闭裁剪）时不切；只切超过 max_error_chars 的错误信息
            if limit and len(error) > self.config.max_error_chars:  # 裁剪开着且错误超长
                result.error = error[: self.config.max_error_chars] + "…(错误信息已裁剪)"  # 截断并加提示后缀
                meta["error_trimmed"] = True   # 告知下游：错误信息被截短过
            result.meta = meta                           # 回写 meta
            return result                                # 失败结果处理完毕，直接返回

        # ---- 分支二：成功结果，按"脱敏 → 转字符串 → 裁剪"处理 ----------
        payload = result.result                          # 取出工具产出本体
        masked = False                                   # 脱敏标记：本次有没有打码
        # 需同时满足四个条件才脱敏：总开关开 / 有身份 / mask_level>0 / 结果是 dict 或 list
        # （字符串无法按字段定位，只有结构化数据才能找到 phone 这类键）
        if self.config.mask_enabled and principal is not None:  # 脱敏总开关开 且 有身份
            level = int(getattr(principal, "mask_level", 0) or 0)  # 取身份的脱敏等级；缺属性/None 兜底 0
            if level > 0 and isinstance(payload, (dict, list)):    # 等级>0 且结果是结构化数据
                payload = self._mask(payload, level)     # 按等级打码（复用企业模块规则）
                masked = True                            # 标记本次已脱敏
                meta["mask_level"] = level      # 记录按哪一级脱敏，便于审计追溯
        # 统一转成字符串后，才能按字符数衡量和裁剪
        text = self._stringify(payload)                  # 结构化转 JSON 文本；字符串原样
        trimmed = False                                  # 裁剪标记：本次有没有被切
        if limit and len(text) > limit:                  # 裁剪开着且文本超长
            # total 取裁剪前长度：模型看到提示就知道"后面还有内容，只是没给我"
            text = text[:limit] + _TRUNCATED_HINT.format(total=len(text), kept=limit)  # 截断 + 拼接裁剪提示
            trimmed = True                               # 标记本次已裁剪
        result.result = text                             # 回写处理后的文本
        # 四个标注位：下游做日志、审计、排障时不用猜测结果被改过什么
        meta["filtered"] = True                          # 标注：结果经过过滤器
        meta["trimmed"] = trimmed                        # 标注：是否被裁剪
        meta["masked"] = masked                          # 标注：是否被脱敏
        meta["chars"] = len(text)                        # 标注：最终字符数
        result.meta = meta                               # 回写 meta
        return result                                    # 返回处理后的结果

    # 【什么】按字段脱敏
    # 【作用】复用企业模块的脱敏规则，不另造一套口径
    # 【为什么】两套脱敏口径必然漂移（企业模块加了新字段规则而这里忘了同步），复用是唯一解
    # 【备注】在函数内部延迟 import：harness 不硬依赖 enterprise 模块，
    #        企业模块不存在时 harness 仍可独立使用
    @staticmethod
    def _mask(payload: Any, level: int) -> Any:
        try:
            from modules.enterprise.masking import mask_record  # 延迟导入企业脱敏模块
            return mask_record(payload, level)           # 按等级打码后返回
        except Exception as e:                           # 模块不存在/打码过程出错
            # 脱敏异常不阻断主流程：记录告警后按原文返回（行为与原有逻辑保持一致）
            _logger.warning(f"结果脱敏失败（按不脱敏处理）: {e}")  # 告警：脱敏失败
            return payload                               # 兜底：原文返回

    # 【什么】把任意结果转成字符串
    # 【作用】回灌上下文时统一是文本；dict/list 用 JSON 保留结构
    # 【为什么】JSON 形态保留了键名结构，模型能认出"这是哪个字段的值"，比 str(dict) 的可读性好
    @staticmethod
    def _stringify(payload: Any) -> str:
        # 字符串原样返回，避免 json.dumps 再给它套上一层引号
        if isinstance(payload, str):                     # 已是字符串
            return payload                               # 原样返回
        if isinstance(payload, (dict, list)):            # 结构化数据
            import json                                  # 局部导入 json（只在需要时用）
            try:
                # ensure_ascii=False：中文直接输出，否则会变成 \uXXXX 转义，又长又难读
                return json.dumps(payload, ensure_ascii=False)  # 转 JSON 文本，中文不转义
            except (TypeError, ValueError):              # 含不可序列化对象（datetime/自定义类）
                # 遇到无法 JSON 序列化的对象（如 datetime、自定义类实例）时退回 str()
                return str(payload)                      # 兜底：str() 转换
        # 数字 / None / 其他对象统一走 str()
        return str(payload)                              # 兜底：str() 转换
