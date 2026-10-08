"""
提示词模板模块
==============
把「提示词」从散落在各模块的字符串常量，升级为可注册、带版本、带变量校验的对象。

【改造前是什么样】提示词原本是 4 处类里的字符串常量 + str.format()：
    ReActAgent.SYSTEM_PROMPT          （角色 + ReAct 流程，变量 max_iter）
    ReActAgent.REASON_PROMPT          （8 个变量拼一整段）
    SummaryGenerator._SUMMARY_PROMPT  （摘要，变量 query/answer）
    SummaryGenerator._MERGE_PROMPT    （合并摘要，变量 summaries）
它们能用，但改一句话要翻文件；少传一个变量就是 KeyError，多传一个变量则静默忽略。

【现在】这 4 处已迁到 builtin.py（正文逐字保留，见 demo 第六幕的回归比对），
    调用方改为 PromptBuilder / registry.render，多拿到变量校验、token 预算裁剪、版本切换。

【本模块提供什么】
    PromptTemplate  一条提示词 = 模板 + 版本号 + 策略标签 + 变量表
    PromptRegistry  按「名字 + 版本」集中存放，支持同名多版本（A/B 对比）
    render()        渲染前先校验变量，缺变量直接抛错，多变量告警并丢弃

【输入 → 输出对照】
    tpl = PromptTemplate(name="qa", template="问题: {q}\n资料: {ctx}")
    tpl.variables            → {"q", "ctx"}
    tpl.render(q="3*7?", ctx="乘法表") → "问题: 3*7?\n资料: 乘法表"
    tpl.render(q="3*7?")     → 抛 PromptError（缺 ctx，不是静默填空）
"""
import string   # 标准库 string：用其 Formatter 解析模板里的 {xxx} 变量占位
from dataclasses import dataclass, field   # dataclass 装饰器与 field：简化类定义、给字段配默认值工厂
from typing import Dict, List, Optional, Set   # 类型注解：字典、列表、可空、集合

from modules.common.logger import get_logger   # 项目统一日志入口
from modules.common.exceptions import AgentException   # 项目统一异常基类，PromptError 要继承它

_logger = get_logger("prompt.template")   # 本模块日志器，带模块名前缀便于按模块过滤日志


# 【什么】提示词模块的专用异常类型
# 【作用】变量缺失/模板未注册/版本不存在都抛它，日志能带上 detail
# 【为什么】项目统一的异常基类是 AgentException，handle_exception 装饰器只对它打印 message/detail 字段，用 ValueError 会丢失 detail
class PromptError(AgentException):   # 提示词专用异常：继承 AgentException 以接入项目统一异常处理
    """提示词相关的异常：变量缺失、模板未注册、版本不存在。

    为什么单独定义一个而不是用 ValueError：
        项目统一的异常基类是 AgentException（exceptions.py:17），
        handle_exception 装饰器只对 AgentException 打印 message/detail 两个字段，
        用 ValueError 会走「未知异常」分支，日志里看不到 detail。
    """   # 文档字符串：解释为何用自定义异常而非内置 ValueError


# 【什么】一条提示词模板对象：正文 + 版本 + 策略标签
# 【作用】让提示词可注册、可按版本回滚，并在渲染前校验变量
# 【为什么】散落的字符串常量无法版本管理和变量校验，封装成对象才能支撑注册表与 A/B 对比
@dataclass   # 用 dataclass 自动生成 __init__/__repr__，字段声明即配置项
class PromptTemplate:   # 提示词模板类：一条提示词的完整描述（正文+版本+策略+说明）
    """一条提示词模板。

    Attributes:
        name:        唯一标识，建议用「模块.用途」命名，如 "react.system"
        template:    模板正文，变量写作 {xxx}；正文里要出现字面花括号须写成 {{ }}
        version:     版本号，同名可注册多个版本用于 A/B 对比
        strategy:    这个模板配套的策略标签（见 strategies.py 的 STRATEGIES）
        description: 一句话说明，给人看（列表展示 / 交接文档用）
    """   # 文档字符串：逐字段说明模板对象的属性含义

    name: str                    # 唯一标识，建议「模块.用途」，如 react.system
    template: str                # 模板正文，变量写 {xxx}；字面花括号要写成 {{ }}
    version: str = "v1"          # 版本号，同名可注册多版做 A/B
    strategy: str = "zero_shot"  # 配套策略标签，见 strategies.STRATEGIES
    description: str = ""        # 一句话说明，给人看（列表展示 / 交接文档）

    # ------------------------------------------------------------------
    # 【变量表是怎么算出来的】用 string.Formatter 遍历模板，取出每个 {xxx} 的字段名。
    #   为什么不用正则 re.findall(r"{(\w+)}")：正则会把 {{ }} 这种转义花括号也算进来，
    #   而 Formatter 遇到 {{ 返回的是字面量 '{'，字段名是 None，正好被下面的 if f 过滤掉。
    #
    # 【处理前 → 处理后】
    #   处理前 template = "问: {query}\n答: {answer}\nJSON 示例: {{\"k\": 1}}"
    #   处理后 variables = {"query", "answer"}      ← 转义的 {{ }} 不算变量
    #
    # 【为什么取 split('.') 和 split('[') 后的第一段】
    #   字段名允许 obj.attr / items[0] 这类写法，前段才是真正要传的变量名。
    # ------------------------------------------------------------------
    # 【什么】模板里所有变量的集合
    # 【作用】渲染前用它比对传参，builder 也靠它判断模板有没有某个插位
    # 【为什么】变量表必须自动从正文解析而非手工维护，否则改正文忘改声明就会校验失真
    @property   # 声明为只读属性：外部用 tpl.variables 像访问字段一样拿到变量集合
    def variables(self) -> Set[str]:   # 变量集合属性：返回模板正文中所有 {xxx} 变量名
        fields: Set[str] = set()   # 初始化空集合，用于收集解析出的变量名（自动去重）
        for _, field_name, _, _ in string.Formatter().parse(self.template):   # 用 Formatter 遍历模板，每次取出一个 {xxx} 的字段名
            if not field_name:   # 字段名为 None 或空串（如 {{ }} 转义花括号）时跳过，不算变量
                continue   # 跳过转义花括号，继续解析下一个占位
            fields.add(field_name.split(".")[0].split("[")[0])   # 取字段名首段：obj.attr→obj、items[0]→items，加入集合
        return fields   # 返回去重后的变量名集合

    # ------------------------------------------------------------------
    # 【渲染：先校验，再 format】
    #   缺变量 → 抛 PromptError。
    #       为什么不能静默填空字符串：prompt 少一段上下文，模型照样会回答，
    #       只是答案质量悄悄变差且不报错 —— 这是最难排查的一类故障（静默降质）。
    #       抛错能在第一次调用时就暴露拼写错误，比如把 rag_context 写成 ragContext。
    #   多变量 → 只告警并丢弃。
    #       为什么这里不抛错：调用方常常一次把整包上下文传进来，
    #       多传几个字段是常态，为它中断调用不划算。
    #
    # 【输入输出对照】
    #   输入 template="问: {query}"；render(query="3*7?", ctx="x")
    #   输出 "问: 3*7?"，并记一条 warning：多余变量 ['ctx']
    #   输入 render()（什么都不传）
    #   输出 抛 PromptError: 模板 react.system 缺少变量: ['query']
    # ------------------------------------------------------------------
    # 【什么】渲染：先校验变量再 str.format
    # 【作用】缺变量立刻抛错避免静默降质；多传的变量只告警并丢弃
    # 【为什么】静默填空会让答案悄悄变差且不留痕迹，抛错能在首次调用时暴露拼写错误
    def render(self, **kwargs) -> str:   # 渲染方法：接收关键字参数，返回替换变量后的最终文本
        need = self.variables   # 取模板需要的变量集合（由 variables 属性自动解析）
        missing = sorted(need - set(kwargs))   # 计算缺失变量：模板需要但调用方没传的，排序让报错信息稳定
        if missing:   # 有缺失变量 → 不能静默填空，必须抛错暴露问题
            raise PromptError(   # 抛出提示词专用异常，message 说明哪个模板缺变量
                f"模板 {self.name}({self.version}) 缺少变量",   # 异常主信息：模板名+版本+缺变量
                f"missing={missing} provided={sorted(kwargs)}",   # 异常明细：缺了哪些、实际传了哪些，便于排查
            )   # raise 语句结束
        extra = sorted(set(kwargs) - need)   # 计算多余变量：调用方传了但模板用不到的
        if extra:   # 有多余变量 → 不抛错（整包传参是常态），但要留痕
            _logger.warning(f"模板 {self.name} 收到多余变量 {extra}，已忽略")   # 记 warning 日志：哪些变量被忽略
            kwargs = {k: v for k, v in kwargs.items() if k in need}   # 过滤掉多余变量，只保留模板需要的，避免 format 时混淆
        return self.template.format(**kwargs)   # 用 str.format 做变量替换，返回最终渲染文本

    # 【什么】导出模板元信息（不含正文）
    # 【作用】给列表接口 / CLI 展示用，避免把长 prompt 打进日志
    # 【为什么】正文可能很长且含业务敏感措辞，展示场景只需要元信息
    def to_dict(self) -> dict:   # 元信息导出方法：返回不含正文的字典，供列表展示和 API 返回
        """模板元信息（列表展示 / API 返回用，不含正文，避免把长 prompt 打到日志里）。"""   # 文档字符串：说明用途与不含正文的原因
        return {   # 组装并返回元信息字典
            "name": self.name,   # 模板唯一标识
            "version": self.version,   # 版本号
            "strategy": self.strategy,   # 配套策略标签
            "variables": sorted(self.variables),   # 变量列表（排序后展示更稳定）
            "description": self.description,   # 一句话说明
        }   # 字典字面量结束


# 【什么】模板注册表：name → version → PromptTemplate 两级字典
# 【作用】集中存放模板，支持同名多版本共存与按版本取用
# 【为什么】一级字典只能覆盖无法回滚，两级结构才能同时保留多版本做 A/B 对比
@dataclass   # 用 dataclass 自动生成 __init__，_items 默认空字典
class PromptRegistry:   # 模板注册表类：管理所有模板的注册、取用、枚举
    """模板注册表：按 name → version → PromptTemplate 两级存放。

    【为什么是两级而不是一级字典】
        一级字典（name → tpl）改提示词只能「覆盖」，改完没法回到上一版，
        也没法同时留两版做 A/B 对比。两级字典允许 v1 / v2 并存，
        取用时默认拿最后注册的那版（dict 保序），也可以显式指定 version。

    【输入 → 输出对照】
        reg.register(PromptTemplate("qa", "问: {q}", version="v1"))
        reg.register(PromptTemplate("qa", "问题: {q}", version="v2"))
        reg.versions("qa")      → ["v1", "v2"]
        reg.get("qa").version   → "v2"            ← 不指定版本取最新
        reg.get("qa", "v1").template → "问: {q}"
    """   # 文档字符串：解释两级字典设计理由与使用示例

    _items: Dict[str, Dict[str, PromptTemplate]] = field(default_factory=dict)   # {模板名: {版本号: 模板对象}}；用 default_factory 避免可变默认参数陷阱

    # ------------------------------------------------------------------
    # 【重复注册怎么处理】同名同版本默认拒绝，必须显式 overwrite=True。
    #   为什么：两个模块各自 register 了同名同版本的模板时，后注册的悄悄盖掉前一个，
    #   线上表现是「改了 A 文件的提示词却没生效」，查起来极其费时。
    #   宁可让第二次注册直接报错，把冲突暴露在启动阶段。
    # ------------------------------------------------------------------
    # 【什么】注册一个模板到两级字典
    # 【作用】同名同版本默认拒绝，必须显式 overwrite=True 才能覆盖
    # 【为什么】静默覆盖会让「改了没生效」的故障极难排查，报错把冲突暴露在启动阶段
    def register(self, tpl: PromptTemplate, overwrite: bool = False) -> None:   # 注册方法：把模板放进两级字典，默认拒绝重复
        versions = self._items.setdefault(tpl.name, {})   # 取该模板名下的版本字典，没有则创建空字典（首次注册这个名字时）
        if tpl.version in versions and not overwrite:   # 同版本已存在且未声明覆盖 → 冲突，拒绝注册
            raise PromptError(   # 抛出异常，阻止静默覆盖
                f"模板 {tpl.name} 的 {tpl.version} 已存在",   # 异常主信息：哪个模板哪个版本重复了
                "如需覆盖请传 overwrite=True",   # 异常明细：告诉调用方如何显式覆盖
            )   # raise 语句结束
        if tpl.version in versions:   # 走到这里且版本已存在 → 说明 overwrite=True，是显式覆盖
            _logger.warning(f"模板 {tpl.name}({tpl.version}) 被覆盖")   # 显式覆盖也要记 warning 留痕，便于审计
        versions[tpl.version] = tpl   # 把模板对象写入两级字典的对应位置
        _logger.debug(f"注册模板: {tpl.name}({tpl.version}) 变量={sorted(tpl.variables)}")   # debug 日志记录注册动作与变量表，便于启动时核对

    # 【什么】按名字（可选版本）取模板
    # 【作用】不指定版本取最后注册那版；名字或版本不存在抛 PromptError
    # 【为什么】取不到模板是配置错误而非业务问题，必须立刻抛错而不是返回 None 让下游二次踩空
    def get(self, name: str, version: str = "") -> PromptTemplate:   # 取模板方法：按名字取，可显式指定版本
        versions = self._items.get(name)   # 取该名字下的版本字典，名字不存在时为 None
        if not versions:   # 名字未注册（None 或空字典）→ 抛错
            raise PromptError(f"模板未注册: {name}", f"已注册: {sorted(self._items)}")   # 报错并回显已注册名单，方便发现拼写错误
        if version:   # 调用方显式指定了版本号
            tpl = versions.get(version)   # 按版本号取模板，版本不存在时为 None
            if tpl is None:   # 指定版本不存在 → 抛错
                raise PromptError(   # 抛出异常
                    f"模板 {name} 没有版本 {version}", f"可用版本: {list(versions)}"   # 报错并回显可用版本列表
                )   # raise 语句结束
            return tpl   # 返回指定版本的模板对象
        return list(versions.values())[-1]          # 不指定版本 → 取最后注册的那版（dict 按插入序保序，最后一个即最新）

    # 【什么】取模板并渲染的一步式写法
    # 【作用】给调用方省一行 get()，变量校验规则与 PromptTemplate.render 一致
    # 【为什么】get+render 是最高频的调用组合，合并成一步降低使用门槛
    def render(self, name: str, version: str = "", **kwargs) -> str:   # 一步式渲染：取模板 + 渲染合并为一次调用
        """取模板并渲染，等价 get(...).render(**kwargs)，给调用方省一行。"""   # 文档字符串：说明等价写法
        return self.get(name, version).render(**kwargs)   # 先取模板再渲染，变量校验规则与 PromptTemplate.render 完全一致

    # 【什么】列出某个模板名下的全部版本
    # 【作用】给 A/B 切换与版本选择接口用，未注册时返回空列表
    # 【为什么】版本枚举是管理接口的基础能力，未注册返回空列表比抛错更适合展示场景
    def versions(self, name: str) -> List[str]:   # 版本枚举方法：返回某名字下所有版本号
        return list(self._items.get(name, {}))   # 取版本字典的键列表；名字未注册时默认空字典 → 返回空列表

    # 【什么】列出全部模板的元信息（按名字排序）
    # 【作用】给管理接口或 CLI 展示用，不含正文避免日志里刷长 prompt
    # 【为什么】展示场景需要稳定排序和轻量数据，正文既长又可能含敏感措辞
    def list_all(self) -> List[dict]:   # 全量元信息方法：返回所有模板的元信息列表
        """列出全部模板元信息，按 name 排序（给 /admin 或 CLI 展示用）。"""   # 文档字符串：说明用途与排序规则
        out: List[dict] = []   # 初始化结果列表
        for name in sorted(self._items):   # 按模板名排序遍历，保证输出顺序稳定
            for tpl in self._items[name].values():   # 遍历该名字下的所有版本（dict 保序，按注册顺序）
                out.append(tpl.to_dict())   # 把每个模板的元信息字典追加进结果
        return out   # 返回全量元信息列表

    # 【什么】列出所有已注册的模板名
    # 【作用】报错信息里回显「已注册: [...]」，也用于快速清点
    # 【为什么】名字清单是最轻量的盘点方式，排序后便于肉眼查找
    def names(self) -> List[str]:   # 名字清单方法：返回所有已注册模板名
        return sorted(self._items)   # 对一级字典的键排序后返回

    # 【什么】取某个模板最新注册的版本号
    # 【作用】展示默认生效版本；模板未注册则返回 None
    # 【为什么】「当前生效的是哪版」是排查问题时第一个要回答的问题
    def latest_version(self, name: str) -> Optional[str]:   # 最新版本方法：返回某名字最后注册的版本号
        versions = self._items.get(name)   # 取该名字下的版本字典，未注册时为 None
        return list(versions)[-1] if versions else None   # 取最后一个键（dict 保序，最后注册即最新）；未注册返回 None
