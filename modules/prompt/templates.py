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
import string
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from modules.common.logger import get_logger
from modules.common.exceptions import AgentException

_logger = get_logger("prompt.template")


class PromptError(AgentException):
    """提示词相关的异常：变量缺失、模板未注册、版本不存在。

    为什么单独定义一个而不是用 ValueError：
        项目统一的异常基类是 AgentException（exceptions.py:17），
        handle_exception 装饰器只对 AgentException 打印 message/detail 两个字段，
        用 ValueError 会走「未知异常」分支，日志里看不到 detail。
    """


@dataclass
class PromptTemplate:
    """一条提示词模板。

    Attributes:
        name:        唯一标识，建议用「模块.用途」命名，如 "react.system"
        template:    模板正文，变量写作 {xxx}；正文里要出现字面花括号须写成 {{ }}
        version:     版本号，同名可注册多个版本用于 A/B 对比
        strategy:    这个模板配套的策略标签（见 strategies.py 的 STRATEGIES）
        description: 一句话说明，给人看（列表展示 / 交接文档用）
    """

    name: str
    template: str
    version: str = "v1"
    strategy: str = "zero_shot"
    description: str = ""

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
    @property
    def variables(self) -> Set[str]:
        fields: Set[str] = set()
        for _, field_name, _, _ in string.Formatter().parse(self.template):
            if not field_name:
                continue
            fields.add(field_name.split(".")[0].split("[")[0])
        return fields

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
    def render(self, **kwargs) -> str:
        need = self.variables
        missing = sorted(need - set(kwargs))
        if missing:
            raise PromptError(
                f"模板 {self.name}({self.version}) 缺少变量",
                f"missing={missing} provided={sorted(kwargs)}",
            )
        extra = sorted(set(kwargs) - need)
        if extra:
            _logger.warning(f"模板 {self.name} 收到多余变量 {extra}，已忽略")
            kwargs = {k: v for k, v in kwargs.items() if k in need}
        return self.template.format(**kwargs)

    def to_dict(self) -> dict:
        """模板元信息（列表展示 / API 返回用，不含正文，避免把长 prompt 打到日志里）。"""
        return {
            "name": self.name,
            "version": self.version,
            "strategy": self.strategy,
            "variables": sorted(self.variables),
            "description": self.description,
        }


@dataclass
class PromptRegistry:
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
    """

    _items: Dict[str, Dict[str, PromptTemplate]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # 【重复注册怎么处理】同名同版本默认拒绝，必须显式 overwrite=True。
    #   为什么：两个模块各自 register 了同名同版本的模板时，后注册的悄悄盖掉前一个，
    #   线上表现是「改了 A 文件的提示词却没生效」，查起来极其费时。
    #   宁可让第二次注册直接报错，把冲突暴露在启动阶段。
    # ------------------------------------------------------------------
    def register(self, tpl: PromptTemplate, overwrite: bool = False) -> None:
        versions = self._items.setdefault(tpl.name, {})
        if tpl.version in versions and not overwrite:
            raise PromptError(
                f"模板 {tpl.name} 的 {tpl.version} 已存在",
                "如需覆盖请传 overwrite=True",
            )
        if tpl.version in versions:
            _logger.warning(f"模板 {tpl.name}({tpl.version}) 被覆盖")
        versions[tpl.version] = tpl
        _logger.debug(f"注册模板: {tpl.name}({tpl.version}) 变量={sorted(tpl.variables)}")

    def get(self, name: str, version: str = "") -> PromptTemplate:
        versions = self._items.get(name)
        if not versions:
            raise PromptError(f"模板未注册: {name}", f"已注册: {sorted(self._items)}")
        if version:
            tpl = versions.get(version)
            if tpl is None:
                raise PromptError(
                    f"模板 {name} 没有版本 {version}", f"可用版本: {list(versions)}"
                )
            return tpl
        return list(versions.values())[-1]          # 不指定版本 → 取最后注册的那版

    def render(self, name: str, version: str = "", **kwargs) -> str:
        """取模板并渲染，等价 get(...).render(**kwargs)，给调用方省一行。"""
        return self.get(name, version).render(**kwargs)

    def versions(self, name: str) -> List[str]:
        return list(self._items.get(name, {}))

    def list_all(self) -> List[dict]:
        """列出全部模板元信息，按 name 排序（给 /admin 或 CLI 展示用）。"""
        out: List[dict] = []
        for name in sorted(self._items):
            for tpl in self._items[name].values():
                out.append(tpl.to_dict())
        return out

    def names(self) -> List[str]:
        return sorted(self._items)

    def latest_version(self, name: str) -> Optional[str]:
        versions = self._items.get(name)
        return list(versions)[-1] if versions else None
