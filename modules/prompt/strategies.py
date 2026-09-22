"""
提示词策略模块
==============
把「怎么问」从模板里拆出来，成为可切换的一层：同一个问题可以走零样本、
少样本、思维链、ReAct 四种问法，切换只改一个字符串参数。

【四种策略各解决什么问题】
    zero_shot  直接问。最省 token，适合事实型、模型本来就答得对的问题。
    few_shot   先给几个问答案例再问。适合格式/口径要求严的场景
               （如"报销单状态要写成 已提交/已审核/已打款 三选一"），
               比在指令里描述十句话更能让模型对齐口径。
    cot        要求先列推理步骤再给结论。适合多步计算、需要中间量的问题；
               代价是输出变长（token 约为 zero_shot 的 2~3 倍）。
    react      按 Reason → Action → Observation 循环，配合工具清单。
               适合"答案不在模型里，得去查/去算"的问题（本项目的默认策略）。

【策略不是越复杂越好】
    zero_shot 能答对的题上 CoT，只是多烧 token；
    react 用在"3*7 等于几"上，模型会真的去调一次计算器（本项目实测过：
    同一句"3*7"被连续调用 9 次，单次对话拖到 14 秒）。
    所以 builder 允许按问题类型切策略，而不是全局一刀切。
"""
from dataclasses import dataclass, field
from typing import List, Optional

from modules.common.logger import get_logger

_logger = get_logger("prompt.strategy")

STRATEGIES = ("zero_shot", "few_shot", "cot", "react")


@dataclass
class FewShotExample:
    """一条少样本示例：问法 + 期望答法。

    tags 用于人工圈定适用范围（如 ["财务","报销"]），
    select() 会先看 tags 有没有命中，命中则加分。
    """

    question: str
    answer: str
    tags: List[str] = field(default_factory=list)


class FewShotStore:
    """少样本示例库：零依赖的相关度检索。

    【为什么不用向量检索】
        示例库一般只有几十条，且都是人工精挑的；为它起一个 embedding 服务
        属于过度设计。这里用字符级 Jaccard 相似度（集合交集 / 并集）排序，
        中文场景下「字面重合度高」与「问法相似」高度相关，成本是零。

    【打分对照（真实数值）】
        query  = "3*7等于几"        字符集 {'3','*','7','等','于','几'}      6 个
        ex1    = "12*8等于多少"     字符集 {'1','2','*','8','等','于','多','少'} 8 个
        交集 = {'*','等','于'} = 3；并集 = 6 + 8 - 3 = 11
        score = 3 / 11 = 0.2727      ← 同类问法，分数最高
        ex2   = "今天天气怎么样"     交集为空 → 0.0
    """

    def __init__(self):
        self._items: List[FewShotExample] = []

    def add(self, question: str, answer: str, tags: Optional[List[str]] = None) -> None:
        self._items.append(FewShotExample(question, answer, tags or []))

    @property
    def size(self) -> int:
        return len(self._items)

    def all(self) -> List[FewShotExample]:
        return list(self._items)

    # ------------------------------------------------------------------
    # 【为什么用「相邻二字组」而不是单字集合】
    #   单字集合会被通用词污染：query "年假怎么算？" 与 "今天天气怎么样？"
    #   共有的只是"怎、么、？"这类谁都有的字，实测得分 0.3，
    #   反而压过了同域的 "报销单状态怎么查？"(0.25) —— 一条无关示例就这么进了 top2。
    #   换成相邻二字组（bigram，"年假" 这种有判别力的片段才起作用）后，
    #   同一句 query 的实测得分变成 [0.1, 0.0, 0.1111, 0.1429]，
    #   对应顺序 报销单状态 / 报销附件 / 天气 / 年假 —— 第一名变成真正相关的那条。
    #
    # 【为什么要先去掉标点】
    #   句尾问号几乎每条示例都有，留着等于给所有示例统一加分，稀释差异。
    #   isalnum() 保留中英文与数字，滤掉标点与空白。
    #
    # 【为什么要 tags 加分】
    #   "报销单状态" 与 "报销流程" 字面重合一般，但都是财务域。
    #   命中 tag 给 0.2 加成，让同域示例排到前面；加成上限 1.0。
    #
    # 【边界】query 或示例为空 → 分母为 0，直接返回 0.0，不除零。
    # ------------------------------------------------------------------
    @staticmethod
    def _grams(text: str) -> set:
        cleaned = "".join(ch for ch in (text or "") if ch.isalnum())
        if len(cleaned) < 2:
            return {cleaned} if cleaned else set()
        return {cleaned[i:i + 2] for i in range(len(cleaned) - 1)}

    @staticmethod
    def similarity(query: str, example: FewShotExample) -> float:
        a = FewShotStore._grams(query)
        b = FewShotStore._grams(example.question)
        if not a or not b:
            return 0.0
        base = len(a & b) / len(a | b)
        bonus = 0.2 if any(t in (query or "") for t in example.tags) else 0.0
        return round(min(1.0, base + bonus), 4)

    def select(self, query: str, k: int = 2, min_score: float = 0.15) -> List[FewShotExample]:
        """挑出最像的 k 条示例；得分低于 min_score 的直接不要（宁缺毋滥）。

        【为什么要 min_score】
            示例库里没有同类问法时，硬塞 2 条不相关的示例反而误导模型。
        【默认 0.15 是怎么定的】
            bigram 相似度在短问句上普遍偏低（实测 0.1~0.35），
            0.15 能拦住"只共用了怎么/吗这类通用词"的伪相关；
            示例库换成自己业务的问法后，这个阈值要按实测分布再调一次。
        """
        scored = sorted(
            ((self.similarity(query, ex), ex) for ex in self._items),
            key=lambda pair: pair[0],
            reverse=True,
        )
        picked = [ex for score, ex in scored[:k] if score >= min_score]
        _logger.debug(
            f"少样本检索: query='{(query or '')[:20]}' 命中 {len(picked)}/{self.size} 条"
        )
        return picked

    def format_block(self, examples: List[FewShotExample]) -> str:
        """把示例渲染成 prompt 片段：编号 + 问 + 答。

        【处理前 → 处理后】
            处理前 [FewShotExample("12*8等于多少", "12*8=96")]
            处理后 "示例 1\n问: 12*8等于多少\n答: 12*8=96"
        """
        if not examples:
            return "（无参考示例）"
        return "\n\n".join(
            f"示例 {i}\n问: {ex.question}\n答: {ex.answer}"
            for i, ex in enumerate(examples, start=1)
        )


def build_strategy_block(
    strategy: str,
    query: str = "",
    store: Optional[FewShotStore] = None,
    k: int = 2,
    tools_desc: str = "",
    max_iter: int = 10,
) -> str:
    """按策略名生成「怎么问」这一段指令。

    Args:
        strategy:   见 STRATEGIES，未知值按 zero_shot 处理并记 warning
        query:      用户问题，few_shot 要用它去示例库检索
        store:      few_shot 的示例库，为空则退化成 zero_shot
        k:          few_shot 取几条示例
        tools_desc: react 的工具清单文本
        max_iter:   react 的循环上限

    Returns:
        一段可直接拼进 system prompt 的指令文本。

    【四种策略返回片段的长度对比（同一句"3*7等于几"实测口径）】
        zero_shot → 1 行，约 20 token
        few_shot  → 示例若干条，约 60~120 token（随 k 变化）
        cot       → 4 行步骤要求，约 60 token
        react     → 三段式 + 工具清单，约 90 token（不含工具清单本身）
    """
    if strategy not in STRATEGIES:
        _logger.warning(f"未知策略 {strategy}，按 zero_shot 处理")

    if strategy == "few_shot":
        # ------------------------------------------------------------------
        # 【示例库为空时为什么要退化而不是报错】
        #   策略是"提示词层"的选择，不该让业务调用失败；
        #   退化成 zero_shot 至少还能答，只是少了口径对齐能力。
        # ------------------------------------------------------------------
        if store is None or store.size == 0:
            _logger.warning("few_shot 策略但示例库为空，退化为 zero_shot")
            return _ZERO_SHOT
        picked = store.select(query, k=k)
        if not picked:
            _logger.warning("示例库无相关示例（低于阈值），退化为 zero_shot")
            return _ZERO_SHOT
        return _FEW_SHOT_HEAD + store.format_block(picked) + _FEW_SHOT_TAIL

    if strategy == "cot":
        return _COT

    if strategy == "react":
        return _REACT_HEAD.format(max_iter=max_iter) + (tools_desc or "（无可用工具）")

    return _ZERO_SHOT


_ZERO_SHOT = "直接给出答案，不要复述问题，不要编造。"

_FEW_SHOT_HEAD = "先参考以下示例的口径与格式，再回答用户问题。\n\n"
_FEW_SHOT_TAIL = "\n\n以上示例仅供参考口径，不要照抄示例中的具体数值。"

_COT = (
    "请按以下步骤回答：\n"
    "1. 列出已知条件\n"
    "2. 写出每一步推导（中间量要写出来，不要跳步）\n"
    "3. 给出最终结论\n"
    "最后一行固定以【结论】开头，只写结论本身。"
)

_REACT_HEAD = (
    "## 工作流程（ReAct）\n"
    "1. Reason：分析当前情况，判断是否需要调用工具\n"
    "2. Action：如需工具，选择合适的工具并确认参数\n"
    "3. Observation：观察工具返回，决定继续调用还是直接作答\n"
    "- 最多循环{max_iter}次，超过后直接给出当前最佳答案\n"
    "- 同一个工具用相同参数重复调用没有意义，看到结果就该作答\n\n"
    "## 可用工具\n"
)
