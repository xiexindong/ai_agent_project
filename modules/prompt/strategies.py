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
from dataclasses import dataclass, field   # dataclass 装饰器与 field：简化示例类的字段声明
from typing import List, Optional   # 类型注解：列表、可空

from modules.common.logger import get_logger   # 项目统一日志入口

_logger = get_logger("prompt.strategy")   # 本模块日志器，带模块名前缀便于按模块过滤日志

STRATEGIES = ("zero_shot", "few_shot", "cot", "react")   # 四种可选策略；未知名按 zero_shot 兜底并记 warning


# 【什么】一条少样本示例：问题 + 期望答复
# 【作用】给 few_shot 策略提供「照着这么答」的口径样例
# 【为什么】示例必须结构化成对象而非裸字符串，才能携带 tags 参与检索打分
@dataclass   # 用 dataclass 自动生成 __init__，三个字段即一条示例的全部信息
class FewShotExample:   # 少样本示例类：一条「问法 + 期望答法」的结构化记录
    """一条少样本示例：问法 + 期望答法。

    tags 用于人工圈定适用范围（如 ["财务","报销"]），
    select() 会先看 tags 有没有命中，命中则加分。
    """   # 文档字符串：说明 tags 的用途与打分规则

    question: str                                          # 示例问题，参与相似度比对
    answer: str                                            # 期望答复，作为口径样例拼进 prompt
    tags: List[str] = field(default_factory=list)          # 领域标签；命中给 0.2 加成，默认空列表


# 【什么】少样本示例库：加入、检索、渲染
# 【作用】给 few_shot 策略提供「挑最像的 k 条」的能力，零外部依赖
# 【为什么】示例库通常只有几十条人工精挑选，上向量检索是过度设计，字符级相似度成本为零
class FewShotStore:   # 少样本示例库类：管理示例的存储、检索、渲染
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
    """   # 文档字符串：解释零依赖检索的设计理由与真实打分示例

    # 【什么】构造空示例库
    # 【作用】示例全放内存列表，检索时对全量打分再排序
    # 【为什么】示例量级小（几十条），全量内存扫描足够快，无需索引结构
    def __init__(self):   # 构造方法：初始化一个空的示例库
        self._items: List[FewShotExample] = []   # 全部示例，顺序即加入顺序

    # 【什么】往库里加一条示例
    # 【作用】tags 不传时记空列表（list 不能做默认参数，只能这样兜）
    # 【为什么】可变对象作默认参数会被所有调用共享，None + 内部兜空列表是标准写法
    def add(self, question: str, answer: str, tags: Optional[List[str]] = None) -> None:   # 添加示例方法：接收问题、答复、可选标签
        self._items.append(FewShotExample(question, answer, tags or []))   # 构造示例对象追加进列表；tags 为 None 时兜成空列表

    # 【什么】示例总条数
    # 【作用】空库判断与日志统计用，避免外部直接摸内部列表
    # 【为什么】封装成只读属性，外部无法绕过接口直接改内部状态
    @property   # 声明为只读属性：外部用 store.size 像访问字段一样拿条数
    def size(self) -> int:   # 示例条数属性：返回库中示例总数
        return len(self._items)   # 返回内部列表长度

    # 【什么】返回全部示例的副本
    # 【作用】调用方拿到的是新列表，改它不影响库内部数据
    # 【为什么】直接返回内部列表引用会让外部无意中污染示例库
    def all(self) -> List[FewShotExample]:   # 全量示例方法：返回所有示例的浅拷贝
        return list(self._items)   # 返回新列表（浅拷贝），外部修改不影响内部

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
    # 【什么】把文本切成相邻二字组（bigram）
    # 【作用】滤掉标点后取 2-gram，短文本（少于 2 字）原样返回单元素集合
    # 【为什么】bigram 比单字更有判别力，"年假"这类片段才能区分同域问法
    @staticmethod   # 静态方法：不依赖实例状态，纯函数式文本切分
    def _grams(text: str) -> set:   # bigram 切分方法：把文本切成相邻二字组集合
        cleaned = "".join(ch for ch in (text or "") if ch.isalnum())   # 滤掉标点与空白：只保留中英文和数字字符（text 为 None 时按空串处理）
        if len(cleaned) < 2:   # 清洗后不足 2 字 → 凑不出 bigram
            return {cleaned} if cleaned else set()   # 有 1 字就返回单元素集合，空串返回空集合（避免下游除零）
        return {cleaned[i:i + 2] for i in range(len(cleaned) - 1)}   # 滑动窗口取相邻二字组，集合法自动去重

    # 【什么】算 query 与一条示例的相关度（0~1）
    # 【作用】Jaccard 为基础分，命中 tags 再加 0.2，上限 1.0，保留 4 位小数
    # 【为什么】字面重合度在中文短问句场景与语义相似高度相关，且计算成本为零
    @staticmethod   # 静态方法：打分逻辑不依赖库实例状态
    def similarity(query: str, example: FewShotExample) -> float:   # 相似度计算方法：返回 0~1 之间的相关度得分
        a = FewShotStore._grams(query)   # 把 query 切成 bigram 集合
        b = FewShotStore._grams(example.question)   # 把示例问题切成 bigram 集合
        if not a or not b:   # 任一为空 → 并集为空，Jaccard 会除零
            return 0.0   # 直接返回 0.0，空文本视为不相关
        base = len(a & b) / len(a | b)   # Jaccard 基础分：交集大小 / 并集大小
        bonus = 0.2 if any(t in (query or "") for t in example.tags) else 0.0   # tags 加成：query 中含任一标签加 0.2，否则加 0
        return round(min(1.0, base + bonus), 4)   # 基础分+加成，封顶 1.0，保留 4 位小数

    # 【什么】挑出最像的若干条示例
    # 【作用】默认取 2 条、门槛 0.15，一条都不够相似就返回空列表
    # 【为什么】没有同类问法时硬塞不相关示例会误导模型，宁缺毋滥
    def select(self, query: str, k: int = 2, min_score: float = 0.15) -> List[FewShotExample]:   # 检索方法：按相关度挑 top-k，低于门槛的丢弃
        """挑出最像的 k 条示例；得分低于 min_score 的直接不要（宁缺毋滥）。

        【为什么要 min_score】
            示例库里没有同类问法时，硬塞 2 条不相关的示例反而误导模型。
        【默认 0.15 是怎么定的】
            bigram 相似度在短问句上普遍偏低（实测 0.1~0.35），
            0.15 能拦住"只共用了怎么/吗这类通用词"的伪相关；
            示例库换成自己业务的问法后，这个阈值要按实测分布再调一次。
        """   # 文档字符串：解释门槛的必要性与默认值由来
        scored = sorted(   # 给每条示例打分并按分数从高到低排序
            ((self.similarity(query, ex), ex) for ex in self._items),   # 生成 (得分, 示例) 元组流：逐条算 query 与示例的相关度
            key=lambda pair: pair[0],   # 排序键取元组第一项：得分
            reverse=True,   # 降序：分数最高的排最前
        )   # sorted 结束，scored 为按得分降序的元组列表
        picked = [ex for score, ex in scored[:k] if score >= min_score]   # 取前 k 条中达标的：低于门槛的宁可不要
        _logger.debug(   # debug 日志记录检索结果，便于调阈值时复盘
            f"少样本检索: query='{(query or '')[:20]}' 命中 {len(picked)}/{self.size} 条"   # 日志内容：query 前 20 字 + 命中数/总数
        )   # _logger.debug 调用结束
        return picked   # 返回挑出的示例列表（可能为空）

    # 【什么】把示例列表渲染成 prompt 片段
    # 【作用】每条编成「示例 N / 问 / 答」，空列表返回"（无参考示例）"
    # 【为什么】统一渲染格式让模型稳定识别示例结构，空列表也要给明确占位避免渲染出空白段
    def format_block(self, examples: List[FewShotExample]) -> str:   # 渲染方法：把示例列表格式化为可拼进 prompt 的文本块
        """把示例渲染成 prompt 片段：编号 + 问 + 答。

        【处理前 → 处理后】
            处理前 [FewShotExample("12*8等于多少", "12*8=96")]
            处理后 "示例 1\n问: 12*8等于多少\n答: 12*8=96"
        """   # 文档字符串：给出渲染前后的对照示例
        if not examples:   # 示例列表为空 → 无示例可渲染
            return "（无参考示例）"   # 返回占位文本，明确告诉模型没有参考示例
        return "\n\n".join(   # 多条示例之间用空行分隔，拼成一整段文本
            f"示例 {i}\n问: {ex.question}\n答: {ex.answer}"   # 单条示例的格式：编号 + 问 + 答
            for i, ex in enumerate(examples, start=1)   # 从 1 开始编号遍历示例列表
        )   # join 结束


# 【什么】生成「怎么问」这段指令文本
# 【作用】按策略名产出对应片段，拼进 system 或模板的 {strategy_block}
# 【为什么】策略必须是独立可替换的一层，调用方改一个字符串就能切换问法
def build_strategy_block(   # 策略块生成函数：本模块对外的唯一入口
    strategy: str,   # 策略名：zero_shot / few_shot / cot / react
    query: str = "",   # 用户问题，few_shot 要用它检索示例
    store: Optional[FewShotStore] = None,   # few_shot 示例库，为空则退化
    k: int = 2,   # few_shot 取几条示例
    tools_desc: str = "",   # react 的工具清单文本
    max_iter: int = 10,   # react 的循环上限
) -> str:   # 返回值：可直接拼进 prompt 的指令文本
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
    """   # 文档字符串：参数说明 + 四种策略的 token 成本对照
    if strategy not in STRATEGIES:   # 策略名不在已知列表 → 配置可能有误
        _logger.warning(f"未知策略 {strategy}，按 zero_shot 处理")   # 记 warning 留痕，但不中断：兜底走 zero_shot

    if strategy == "few_shot":   # 少样本策略分支
        # ------------------------------------------------------------------
        # 【示例库为空时为什么要退化而不是报错】
        #   策略是"提示词层"的选择，不该让业务调用失败；
        #   退化成 zero_shot 至少还能答，只是少了口径对齐能力。
        # ------------------------------------------------------------------
        if store is None or store.size == 0:   # 示例库没传或为空 → 无法提供示例
            _logger.warning("few_shot 策略但示例库为空，退化为 zero_shot")   # 记 warning：配置与资源不匹配
            return _ZERO_SHOT   # 退化为零样本指令，保证业务调用不失败
        picked = store.select(query, k=k)   # 用 query 检索最相关的 k 条示例
        if not picked:   # 一条都不达标（全低于阈值）→ 无相关示例可用
            _logger.warning("示例库无相关示例（低于阈值），退化为 zero_shot")   # 记 warning：示例库覆盖不到这类问法
            return _ZERO_SHOT   # 退化为零样本指令，宁缺毋滥
        return _FEW_SHOT_HEAD + store.format_block(picked) + _FEW_SHOT_TAIL   # 拼接完整 few_shot 指令：头 + 示例块 + 尾

    if strategy == "cot":   # 思维链策略分支
        return _COT   # 返回固定的 CoT 步骤要求文本

    if strategy == "react":   # ReAct 策略分支
        return _REACT_HEAD.format(max_iter=max_iter) + (tools_desc or "（无可用工具）")   # 填循环上限 + 拼工具清单；清单为空时给明确占位

    return _ZERO_SHOT   # 默认/兜底分支：zero_shot 或未知策略都返回零样本指令


_ZERO_SHOT = "直接给出答案，不要复述问题，不要编造。"   # zero_shot 正文：一行，约 20 token

_FEW_SHOT_HEAD = "先参考以下示例的口径与格式，再回答用户问题。\n\n"   # few_shot 头部：说明示例的用途是参考口径
_FEW_SHOT_TAIL = "\n\n以上示例仅供参考口径，不要照抄示例中的具体数值。"   # few_shot 尾部：防止模型照抄示例数值

_COT = (   # cot 正文：固定的四行步骤要求
    "请按以下步骤回答：\n"   # 步骤总起句
    "1. 列出已知条件\n"   # 第一步：列出已知条件
    "2. 写出每一步推导（中间量要写出来，不要跳步）\n"   # 第二步：逐步推导，不许跳步
    "3. 给出最终结论\n"   # 第三步：给出结论
    "最后一行固定以【结论】开头，只写结论本身。"   # 输出格式约定：结论行以【结论】开头，便于程序提取
)   # 多行字符串结束，_COT 定义完成

_REACT_HEAD = (   # react 正文头部：流程说明 + 规则 + 工具清单标题（工具清单由调用方拼接）
    "## 工作流程（ReAct）\n"   # 流程小节标题
    "1. Reason：分析当前情况，判断是否需要调用工具\n"   # 步骤一：推理判断
    "2. Action：如需工具，选择合适的工具并确认参数\n"   # 步骤二：选工具并确认参数
    "3. Observation：观察工具返回，决定继续调用还是直接作答\n"   # 步骤三：观察并决策
    "- 最多循环{max_iter}次，超过后直接给出当前最佳答案\n"   # 规则一：循环上限（max_iter 由调用方填入）
    "- 同一个工具用相同参数重复调用没有意义，看到结果就该作答\n\n"   # 规则二：禁止同参重复调用（针对实测的 9 次重复调用问题）
    "## 可用工具\n"   # 工具清单标题；清单正文由调用方拼在后面
)   # 多行字符串结束，_REACT_HEAD 定义完成
