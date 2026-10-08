"""
记忆管理模块
=============
企业级智能向量记忆系统核心：短期记忆 + 长期记忆双层架构。
- 短期记忆：最近N轮完整对话原文（精准上下文）
- 长期记忆：PG pgvector 摘要向量语义召回（跨会话历史匹配）
覆盖边界4（上下文溢出自动压缩）、边界8（记忆召回阈值过滤）。
"""
# 【什么】标准库与项目内依赖
# 【作用】re 做关键词切分、check_token_overflow 判压缩、PGManager 取数、get_settings 读阈值
import re                                                     # 正则库：入库筛选与关键词校验里做标点/空白清理、n-gram 切分
from datetime import datetime                                 # 取当前时刻，算记忆距今天数，喂给时效衰减排序
from typing import List, Optional                             # 类型标注工具（当前签名未直接用到，保留以备扩展）
from modules.common.logger import get_logger                  # 统一日志工厂，产出带 "memory" 前缀的 logger
from modules.common.exceptions import handle_exception        # 异常兜底装饰器：召回抛错时返回默认值 []，不中断对话
from modules.common.token_counter import count_tokens, check_token_overflow  # token 计数与溢出判定（边界4 压缩的触发依据）
from modules.memory.pg_manager import PGManager               # PG 读写封装：会话/历史/工具流水全部走它
from modules.memory.embedding_service import get_embedding_service  # 问句转向量的服务单例获取入口
from modules.memory.summary_generator import SummaryGenerator  # 摘要合并器：压缩时把多条久远摘要合成一条
from config import get_settings                               # 全局配置：阈值与治理开关统一从这儿读

_logger = get_logger("memory")                      # 模块级日志器，前缀 memory


# 【什么】双层记忆管理器（短期原文 + 长期向量召回）
# 【作用】对外只暴露 save_turn 存一轮、build_context 拼上下文两个入口
# 【为什么】把 PG、向量、摘要三个下游组件封装成一张脸，上层（ReAct）无需感知存储细节
class MemoryManager:
    """双层记忆管理器：短期原文 + 长期向量召回。"""

    # 【什么】组装三个下游组件并把 3 个阈值固化成实例属性
    # 【作用】运行期不再读配置，改配置需重建实例才生效
    # 【为什么】阈值在运行期保持不变，避免同一轮对话里前后两轮判定口径漂移
    def __init__(self, llm_client=None):
        self.pg = PGManager()                                   # PG 读写：会话 / 记忆 / 工具流水
        self.embedding_service = get_embedding_service()        # 问句转向量，召回前用
        self.summary_generator = SummaryGenerator(llm_client=llm_client)   # 压缩时合并多条摘要
        s = get_settings()                                      # 读一次全局配置，下面三个阈值与治理开关都从它取
        self._max_history_tokens = s.max_history_tokens         # 上下文 token 上限，默认 4000，超了触发压缩
        self._short_rounds = s.short_memory_rounds              # 短期记忆轮数，默认 3 轮
        self._similarity_threshold = s.similarity_threshold     # 召回相似度下限，默认 0.65，低于它丢弃
        self._low_confidence = False                            # 本次召回是否低置信，供上层提示人工审核
        # ------------------------------------------------------------------
        # 【记忆治理三阈值】入库筛选 / 冲突处理 / 时效衰减的开关与参数
        #   _filter_enabled     : False 时 save_turn 不过滤（行为回到改造前）
        #   _conflict_threshold : 新旧记忆相似度 ≥ 它判为同一主题（默认 0.90）
        #   _half_life_days     : 时效半衰期，默认 180 天——旧记忆排序权重每 180 天减半
        # ------------------------------------------------------------------
        self._filter_enabled = s.memory_filter_enabled          # 寒暄/过短问句不入库，True 默认开
        self._conflict_threshold = s.memory_conflict_threshold  # 冲突判定阈值，默认 0.90
        self._half_life_days = s.memory_recency_half_life_days  # 时效半衰期天数，默认 180

    # 【什么】保存一轮对话的全量数据（带入库筛选与冲突处理）
    # 【作用】原文、摘要、摘要向量、问句向量、工具日志一次落库，缺任一项后续记忆能力会残废
    # 【为什么】入库前先筛选、入库后再废旧，保证长期记忆库只进"值得记"的内容且同主题只有最新一条
    def save_turn(
        self,
        session_id: str,                        # 会话号，定位"哪本笔记本"
        user_id: str,                           # 用户号，多用户记忆隔离依据
        user_query: str,                        # 用户这一轮的原话
        llm_full_answer: str,                   # 完整原文，短期记忆直接读它
        llm_summary: str,                       # 一句话摘要，上下文投喂用
        summary_embedding: list,                # 摘要向量（1024 维），长期记忆主检索列
        tool_call_log: list | None = None,      # 本轮用过的工具流水，None 落库为 []
        rag_sources: list | None = None,        # 本轮引用的知识库来源，None 落库为 []
        query_embedding: list | None = None,    # 问句向量，当前不参与检索，预留给重复提问检测
    ) -> int | None:
        """
        保存一轮对话全量数据（入库筛选 → 落库 → 冲突取代）。

        【治理流水线，三步】
            第1步 - 入库筛选（_should_memorize）：
                寒暄 / 过短 / 纯标点的问句直接跳过，返回 None 不落库
            第2步 - 落库（pg.save_history）：
                原文 + 摘要 + 双向量 + 工具日志一次写入，拿回新行主键
            第3步 - 冲突处理（find_conflict_memory → supersede_memory）：
                新记忆入库后，把同用户相似度 ≥ 0.90 的旧记忆置 superseded

        【输入输出对照】
            筛选拦截：save_turn(..., user_query="你好", ...)  → None（不落库）
            正常入库：save_turn(..., user_query="差旅报销标准是多少", ...)
                      → 42（新行 id）
            冲突取代：库里已有 id=40 "用户住址:上海"(active, 相似度0.93)，
                      本次存 "用户住址:北京" → 新行 id=42 入库，
                      旧 id=40 置 memory_status='superseded', superseded_by=42
        """
        # ------------------------------------------------------------------
        # 【第1步·入库筛选】不值得进长期记忆的内容在这里挡下。
        #   【不滤会怎样】"你好""好的""哈哈哈"这类寒暄全部进库，
        #   长期记忆库被噪声灌满后，召回 top-5 里寒暄挤掉真正有用的历史。
        # ------------------------------------------------------------------
        if self._filter_enabled and not self._should_memorize(user_query, llm_summary):  # 开关开着且筛选不通过才拦截；开关关着时行为回到改造前全量入库
            _logger.info(f"问句未通过入库筛选，不写入长期记忆: '{user_query[:20]}'")     # 打日志留痕，问句截前 20 字防止日志爆长
            return None                                           # 返回 None 明确告诉上层"这轮没存"，与新行 id 区分

        # 【第2步·落库】先拿到新行主键，第 3 步取代旧记忆要用它回填 superseded_by
        history_id = self.pg.save_history(                        # 一次写入全量字段，返回新行自增主键
            session_id, user_id,                                  # 会话归属 + 用户归属（隔离边界）
            user_query, llm_full_answer, llm_summary,             # 原文三件套：问、答、摘要
            summary_embedding, tool_call_log, rag_sources,        # 摘要向量 + 工具流水 + 知识来源
            query_embedding,                                      # 问句向量（预留，当前不参与检索）
        )

        # ------------------------------------------------------------------
        # 【第3步·冲突处理】新记忆已入库（id 已知）再取代旧的。
        #   【顺序为什么是"先插新、再废旧"】反过来（先废旧再插新）时万一
        #   save_history 抛异常，旧记忆已被置 superseded 而新的没进去——
        #   这个主题的记忆凭空消失；先插新最坏情况也只是两条并存一轮。
        # ------------------------------------------------------------------
        if history_id and summary_embedding:                      # 两个前置：落库成功拿到 id + 有摘要向量才算相似度
            conflict = self.pg.find_conflict_memory(              # 在同用户 active 记忆里找相似度 ≥ 0.90 的旧记录
                summary_embedding, user_id, self._conflict_threshold,  # 传入新摘要向量、用户号、冲突阈值
            )
            if conflict:                                          # 找到同主题旧记忆才执行取代，找不到直接跳过
                self.pg.supersede_memory(conflict["id"], history_id)   # 旧记录置 superseded 并回填 superseded_by=新 id
                _logger.info(                                     # 留审计日志：谁取代了谁、相似度多少
                    f"记忆冲突处理: 新记忆 id={history_id} 取代旧记忆 "
                    f"id={conflict['id']} (相似度 {conflict['similarity']})"
                )
        return history_id                                         # 返回新行 id（被拦截时已提前返回 None）

    # 【什么】入库筛选：判断一轮对话值不值得进长期记忆
    # 【作用】挡住寒暄、过短、纯标点三类噪声；规则不依赖 LLM，零成本零延迟
    # 【为什么】长期记忆库的信噪比直接决定召回质量，必须在入口侧用廉价规则挡住噪声
    @staticmethod
    def _should_memorize(user_query: str, llm_summary: str = "") -> bool:
        """
        入库筛选（规则版）：三关全过才返回 True。

        【三关规则】
            ① 长度关：问句去空白后 < 4 字符 → 拦（"好的""嗯""ok" 都过不了）
            ② 寒暄关：整句命中寒暄/告别/致谢词表 → 拦（"你好""谢谢""再见"）
            ③ 内容关：问句去标点后是空的 → 拦（"？？？""！！"）

        【判定例子】
            "你好"                        → False（命中寒暄词表）
            "好的"                        → False（长度 2 < 4）
            "差旅报销标准是多少"            → True
            "帮我算一下12*8"              → True
            "？？？"                      → False（去标点后为空）
            "你好，请问报销流程"           → True（不是整句寒暄，是带正事的问句）

        【为什么整句匹配而不是包含匹配】"你好，请问报销流程"里也有"你好"，
            按包含匹配会被误杀；寒暄当"开场白"整句才算噪声，混在句中是正常表达。
        """
        q = (user_query or "").strip()                            # None 兜底为空串再去首尾空白，后面统一按 str 处理
        if len(q) < 4:
            return False                                       # ① 长度关
        if re.sub(r"[，。！？!?.,~～\s]", "", q) == "":
            return False                                       # ③ 内容关（纯标点/空白）
        # ② 寒暄关：整句（去标点空白后）等值命中才算寒暄
        normalized = re.sub(r"[，。！？!?.,~～\s]", "", q).lower()  # 去标点空白并转小写，得到规范化后的整句用于等值比对
        chitchat = {
            "你好", "您好", "嗨", "哈喽", "hello", "hi",
            "谢谢", "多谢", "感谢", "thanks", "thankyou",
            "再见", "拜拜", "bye", "好的", "ok", "嗯嗯", "哈哈",
            "明白了", "知道了", "收到",
        }                                                         # 寒暄/致谢/告别/敷衍词表，命中即视为无记忆价值
        return normalized not in chitchat                         # 不在词表 = 有实质内容 → True；命中 → False 拦截

    # 【什么】构建塞给 LLM 的记忆上下文
    # 【作用】长期召回 + 短期原文拼成一段文本，超 token 就压缩，返回 dict 供 prompt 直接取用
    # 【为什么】上下文是 LLM 唯一能看到"过去"的窗口，双层拼接兼顾精准（短期原文）与广度（长期语义召回）
    def build_context(
        self,
        session_id: str,                        # 会话号，取短期记忆用
        user_id: str,                           # 用户号，长期召回限定在此用户内
        query: str,                             # 本轮提问，用于召回与最终拼接展示
        query_embedding: list | None = None,    # 传了就省一次 embed 调用
    ) -> dict:
        """
        构建塞给 LLM 的记忆上下文 = 长期向量召回 + 短期原文 + 压缩标记。

        【双层记忆为什么这么设计？】
            短期记忆（最近 3 轮原文）：精准但容量小，撑不住跨会话历史
            长期记忆（向量召回历史摘要）：能跨会话语义匹配，但是"压缩版摘要"
            两层拼起来既精准又广覆盖。

        【返回 dict 的三个键 + 输入输出对比】
            输入: session_id="abc", user_id="cli-user", query="3*7"
            输出:
                {
                  "long_term_memory":  [{"summary":"3*5=15", "similarity":0.91}, ...],
                  "short_term_memory":[{"query":"3*5?", "answer":"3*5=15"}, ...],
                  "context_text": "【历史记忆摘要】\n- 3*5=15\n\n【近期对话】\n用户: 3*5?\nAI: 3*5=15",
                  "compressed": False
                }
                → context_text 直接拼到 ReAct reason prompt 的 memory_context 字段
                → compressed=True 时表示发生过 token 压缩（前端可提示"历史已压缩"）

        【覆盖的安全边界】
            边界4 - token 阈值压缩：上下文超 max_history_tokens 时调
                    _compress_long_term 把多条久远摘要合成一条
            边界8 - 相似度过滤：长期召回结果低于 similarity_threshold 舍弃，
                    避免"用户问天气、召回到股票"的误匹配
        """
        long_term = self._recall_long_term(query, user_id, query_embedding)   # 长期记忆：向量召回 + 阈值过滤 + 时效重排

        short_term = self.pg.get_short_term_history(              # 短期记忆：本会话最近 N 轮原文（默认 3 轮）
            session_id, user_id, self._short_rounds
        )

        context_text = self._format_context(long_term, short_term, query)   # 两层记忆拼成一段纯文本，准备塞进 prompt
        compressed = False                                        # 压缩标记：默认未压缩，触发边界4 后改 True
        # 这个地方 有一个疑问 为什么要熔断
        # 【设计意图·边界4 溢出压缩】先拼好再量 token，超限就把多条久远摘要压成一条重拼；
        #   不压缩的话 prompt 会越来越长，最终顶爆模型上下文窗口直接报错
        if check_token_overflow(context_text, self._max_history_tokens):   # 当前拼接文本 token 数是否超过上限（默认 4000）
            _logger.warning("记忆上下文溢出，触发压缩（边界4）")              # warning 级：非常态路径，需要被监控看见
            long_term = self._compress_long_term(long_term, user_id, query)  # 多条久远摘要合并成一条全局摘要
            context_text = self._format_context(long_term, short_term, query)  # 用压缩后的长期记忆重拼上下文文本
            compressed = True                                     # 置压缩标记，供上层/前端提示"历史已压缩"

        return {
            "long_term_memory": long_term,                        # 长期记忆列表（召回明细，含相似度），供调试/展示
            "short_term_memory": short_term,                      # 短期记忆列表（最近几轮原文），供调试/展示
            "context_text": context_text,                         # 拼好的纯文本，ReAct prompt 直接取用
            "compressed": compressed,                             # 是否发生过 token 压缩
            "low_confidence": self._low_confidence,               # 召回是否低置信（字面全不重合），供上层触发人工审核
        }

    # 【什么】长期记忆召回：向量检索 + 相似度过滤 + 关键词二次校验
    # 【作用】跨会话找出语义相近的历史摘要；异常返回 []，不中断整个对话
    # 【为什么】记忆是增强不是命脉——召回挂了对话还得继续，所以用装饰器兜底返回空列表
    @handle_exception(default=[], log_msg="长期记忆召回异常")
    def _recall_long_term(
        self, query: str, user_id: str, query_embedding: list | None = None
    ) -> list:
        """
        用问题向量召回历史摘要，并按相似度阈值过滤。

        【两步流水线】
            第1步 - 向量召回（pg.search_memory_by_vector top_k=5）：
                SQL 用余弦距离排序，返回前 5 条"语义最相近"的记录
            第2步 - 相似度过滤（边界8）：
                similarity < similarity_threshold 的记录丢弃

        【过滤前后对比】
            召回结果（5 条，含分数）：
                [
                    {"summary":"3*5=15", "similarity":0.91},  ✓ 保留
                    {"summary":"4*6=24", "similarity":0.82},  ✓ 保留
                    {"summary":"用户问天气", "similarity":0.45}, ✗ 丢弃（<0.6 阈值）
                    {"summary":"股票分析", "similarity":0.38},  ✗ 丢弃
                    {"summary":"美食推荐", "similarity":0.21},  ✗ 丢弃
                ]
            过滤后（2 条）：
                [{"summary":"3*5=15", "similarity":0.91},
                 {"summary":"4*6=24", "similarity":0.82}]

        【阈值的作用】用户问数学，但向量库里有"天气""股票""美食"等无关历史，
            不滤掉的话塞给 LLM 的上下文会被噪声污染，回答质量下滑。
        """
        if query_embedding is None:                               # 上层没传向量才现算；传了直接用，省一次 embed 调用
            query_embedding = self.embedding_service.embed(query)  # 把本轮问句转成 1024 维向量



        results = self.pg.search_memory_by_vector(
            query_embedding, user_id, top_k=5   # 先取语义最近的 5 条，再按相似度阈值与字面重合过滤
        )
        filtered = [
            r for r in results
            if r["similarity"] >= self._similarity_threshold      # 边界8：低于阈值（默认 0.65）的召回直接丢弃
        ]
        if results and not filtered:                              # 有召回但全被阈值滤掉时打日志，便于排查"为什么没记忆"
            _logger.info(
                f"历史记忆召回{len(results)}条但全部低于阈值{self._similarity_threshold}，已舍弃"
            )

        # ------------------------------------------------------------------
        # 【关键词二次校验】向量说"语义像"，再问一句"字面有没有交集"
        #
        # 【为什么要这个？】向量召回只看余弦距离，存在"语义近但字面完全不同"的
        #   误匹配（问"报销流程"召回"差旅审批怎么走"）。在相似度阈值之外再加一道
        #   字面复核（中文 2-gram / 英文按词），两边有共同词块才保留 = 双保险。
        #
        # 【对比例子】query = "公司的报销流程是什么"
        #   ✓ "报销流程需要提交发票"  交集 {报销 销流 流程} → 保留
        #   ✗ "差旅审批怎么走"        交集 {}              → 丢弃
        #
        # 【本项目说明·三种结果走向】
        #   全部命中 → 原样保留；部分命中 → 丢掉无重合的并打日志；
        #   全部不中 → 不清空（相似度明明过阈值，可能只是措辞差异），
        #              保留原结果 + 置 _low_confidence=True，交上层触发人工审核
        # ------------------------------------------------------------------
        self._low_confidence = False                          # 每次召回先复位：上一轮的低置信标记不能带进这一轮
        # 两个前置条件：过了相似度阈值的结果非空 + 配置开关 memory_keyword_check 打开
        if filtered and get_settings().memory_keyword_check:
            # 列表推导逐条做字面复核，通过的收进 checked
            checked = [
                r for r in filtered
                # 优先比 summary；摘要缺失时用 or 短路退回原始问句 query（两个都没有时 _has_keyword_overlap 内部按空串处理）
                if self._has_keyword_overlap(query, r.get("summary", "") or r.get("query", ""))
            ]
            if checked:                                       # 至少一条字面重合 → 走"精筛"分支
                if len(checked) < len(filtered):              # 只在部分被丢时打日志；全部通过不刷屏
                    _logger.info(
                        f"记忆关键词二次校验: 丢弃 {len(filtered) - len(checked)} 条无字面重合的召回"
                    )
                filtered = checked                            # 用精筛结果替换原召回
            else:
                self._low_confidence = True                   # 全部无字面重合：标低置信，filtered 保留不做清空
                _logger.info("记忆召回字面完全不重合，标记为低置信（触发人工审核）")  # 写明标记原因，排查"这条为什么进了人工审核"

        _logger.info(f"长期记忆召回: 命中={len(filtered)}/{len(results)}")  # 最终命中数/原始召回数，一行观测收口

        # ------------------------------------------------------------------
        # 【时效衰减重排】similarity 只管"语义像不像"，不管"过没过期"。
        #   有效分 = similarity × 0.5^(天数/半衰期)，默认半衰期 180 天：
        #     今天写的记忆   天数=0    → 衰减系数 1.00（0.91 → 0.910）
        #     180 天前的记忆  天数=180  → 衰减系数 0.50（0.91 → 0.455）
        #     360 天前的记忆  天数=360  → 衰减系数 0.25（0.91 → 0.228）
        #
        #   【作用】"去年说的偏好"与"上个月说的偏好"语义分相同时，新者优先。
        #   【只影响排序不动 similarity 字段】阈值过滤（0.65）已经在前面做完了，
        #   这里若把衰减后的分写回 similarity，会把刚过阈值的记忆误判成低置信。
        # ------------------------------------------------------------------
        if self._half_life_days > 0:                           # 开关：半衰期=0 时整段跳过，保持召回原顺序
            now = datetime.now()                               # 全程只取一次当前时刻，所有条目共用同一基准

            # 【什么】sort 的 key 函数：把一条记忆换算成"相似度 × 时效系数"
            # 【为什么】时间字段可能缺失或异常，全部兜底按 0 天（新记忆）处理——宁可不奖励，不可误惩罚
            def _effective(item: dict) -> float:               # sort 的 key：把一条记忆换算成"相似度 × 时效系数"
                try:
                    t = item.get("time")                       # 记忆创建时间；可能缺失(None)或不是 datetime
                    days = (now - t).days if t else 0          # 有时间算距今天数；缺失按 0 天 = 不衰减
                except Exception:
                    days = 0                                    # 时间字段异常按新记忆算，不惩罚
                decay = 0.5 ** (max(days, 0) / self._half_life_days)  # 每过一个半衰期系数减半；max 防未来日期算出负天数
                return item["similarity"] * decay              # 有效分只喂给排序，不写回 similarity 字段

            filtered.sort(key=_effective, reverse=True)        # 按有效分降序：又新又像的排前面
            _logger.debug(f"记忆召回按时效衰减重排完成: {len(filtered)} 条")  # debug 级：常规召回不刷屏

        return filtered                                         # 返回最终召回列表：过滤 + 精筛 + 时效重排后的结果

    # 【什么】关键词二次校验（静态方法，不依赖实例状态）
    # 【作用】向量相似之外再看字面是否重合，两边有交集才放行这条召回
    # 【为什么】纯靠余弦距离会有"语义近但字面无关"的误匹配，字面复核是召回质量的第二道保险
    @staticmethod
    def _has_keyword_overlap(query: str, text: str) -> bool:
        """
        关键词二次校验：中文按 2-gram、英文按词，判定问句与历史摘要是否有字面重合。

        【作用】向量召回只保证「语义相近」，可能召回「语义像但字面无关」的历史；
        这里做字面复核，两边字块有交集才放行。调用处见第 278-297 行，开关是
        memory_keyword_check；若全部被否掉，会把 _low_confidence 置 True 触发人工审核。

        【实跑：输入 → 切分中间值 → 判定】

        ① 中文·有重合（含汉字即走 2-gram：去空白后相邻两字滑窗）
             query "公司的报销流程是什么" → {公司 司的 的报 报销 销流 流程 程是 是什 什么}
             text  "报销流程需要提交发票" → {报销 销流 流程 程需 需要 要提 提交 交发 发票}
             交集 {报销 销流 流程} → True

        ② 中文·无重合
             text  "今天天气怎么样" → {今天 天天 天气 气怎 怎么 么样}
             交集 {} → False

        ③ 英文·有重合（无汉字即走分词）
             注意第 372 行已先删掉所有空白，句子会粘成一整串，分词退化成「整句当一个词」：
             query "how to reset password"  → {howtoresetpassword}
             text  "reset your password here" → {resetyourpasswordhere}
             交集 {} → False     ← 两边都含 reset、password，却判「不重合」
             只有「去掉空格后完全相同」才为 True，等于退化成整字符串比对。

        ④ 中英混合（只要含一个汉字，整段都走 2-gram）
             query "Python 怎么读文件" → {py yt th ho on n怎 怎么 么读 读文 文件}
             ← "Python" 被切成字母对 py/yt/th/ho/on，不再是一个词。
             同一对英文词，在纯英文句里按词比、在混了汉字的句里按字母对比，两套口径。

        ⑤ 边界情况
             query "好" → range(len-1) = range(0) → 切出空集合 → 永远 False（单字查询必丢）
             query "报销，流程" → {报销 销， ，流 流程} → 第 372 行只删空白不删标点，
                   标点会切断本该命中的 2-gram（此处"销流"被逗号切没了）。

        【结论】③ 的后果最严重：本校验对英文几乎恒为 False。英文提问时召回的记忆
        会被整批丢弃、并触发人工审核 —— 全程不报错，只是英文用户的长期记忆从不生效。
        修法两处：先分词再删空白（即调换第 372 与 375 行的顺序），以及中英分开处理，
        不要「含一个汉字」就整段按 2-gram 切。
        """
        # 【什么】内部切分函数：把一句话切成可比对的字块集合
        # 【作用】中文走相邻两字滑窗（2-gram），英文走按词切分，输出集合供交集运算
        def grams(s: str) -> set:
            s = re.sub(r"\s+", "", s.lower())                   # 转小写后删除所有空白字符（含空格/制表/换行）
            if re.search(r"[\u4e00-\u9fff]", s):                # 只要含一个汉字，整段按中文处理走 2-gram
                return {s[i:i + 2] for i in range(len(s) - 1)}  # 相邻两字滑窗切 2-gram 集合；单字切出空集合
            return set(re.findall(r"[a-z0-9]+", s))             # 纯英文按字母数字连续段分词，转集合

        return bool(grams(query) & grams(text))                 # 两个字块集合求交集，非空即 True（有字面重合）

    # 【什么】上下文压缩：多条久远摘要合成一条
    # 【作用】超 token 上限时的兜底手段；合并失败就原样返回，宁可超也不丢内容
    # 【为什么】压缩是为保对话不中断，不是为省 token 而冒险丢历史——合并失败时原样返回是安全的下策
    def _compress_long_term(self, long_term: list, user_id: str, query: str) -> list:
        """
        上下文压缩：将多个久远摘要合并为一个全局摘要，减少 token 占用。
        """
        if len(long_term) <= 1:                                   # 只有 0/1 条时无可压缩，直接原样返回
            return long_term
        summaries = [item["summary"] for item in long_term]       # 抽出所有摘要文本，准备交给合并器
        merged = self.summary_generator.merge_summaries(summaries)  # 调 LLM 把多条摘要合成一条全局摘要
        if not merged:                                            # 合并失败（LLM 异常/返回空）时兜底
            return long_term                                      # 宁可上下文超 token 也不丢历史内容
        return [{
            "summary": merged,                                    # 合并后的全局摘要，替代原来的多条
            "query": query,                                       # 挂本轮问句，保持返回结构字段完整
            "answer": merged,                                     # answer 字段同摘要，保持结构字段完整
            "similarity": 1.0,                                    # 合成条目视为满分相似，避免下游再次过滤误伤
            "time": "merged",                                     # 时间字段占位标记，标识这是合成条目而非真实历史
            "merged": True,                                       # 合并标记，下游可识别"这条是压出来的"
        }]

    # 【什么】把两层记忆拼成一段给 LLM 看的文本
    # 【作用】按「历史摘要 → 近期对话」顺序输出，空的部分整块跳过，不留空标题
    # 【为什么】LLM 只看文本不看结构；先远后近的顺序让模型先建立背景、再衔接最近对话流
    @staticmethod
    def _format_context(long_term: list, short_term: list, query: str) -> str:
        """拼接记忆上下文为文本。"""
        blocks = []                                               # 收集所有文本行，最后一次 join 成整段
        if long_term:                                             # 长期记忆非空才输出标题块，避免留空标题
            blocks.append("【历史记忆摘要】")                      # 长期记忆区块标题
            for item in long_term:                                # 逐条召回摘要
                blocks.append(f"- {item.get('summary', '')}")     # 每条一行短横线列表；summary 缺失兜底空串
        if short_term:                                            # 短期记忆非空才输出标题块
            blocks.append("\n【近期对话】")                        # 短期记忆区块标题，前导空行与上一块隔开
            for item in short_term:                               # 逐条近期对话
                blocks.append(f"用户: {item.get('query', '')}")   # 用户问句原样输出
                blocks.append(f"AI: {item.get('answer', '')[:200]}")    # 回答只取前 200 字，长答案省 token
        return "\n".join(blocks)                                  # 所有行用换行连接；两层都空时返回空串
