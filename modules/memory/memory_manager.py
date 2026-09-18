"""
记忆管理模块
=============
企业级智能向量记忆系统核心：短期记忆 + 长期记忆双层架构。
- 短期记忆：最近N轮完整对话原文（精准上下文）
- 长期记忆：PG pgvector 摘要向量语义召回（跨会话历史匹配）
覆盖边界4（上下文溢出自动压缩）、边界8（记忆召回阈值过滤）。
"""
from typing import List, Optional
from modules.common.logger import get_logger
from modules.common.exceptions import handle_exception
from modules.common.token_counter import count_tokens, check_token_overflow
from modules.memory.pg_manager import PGManager
from modules.memory.embedding_service import get_embedding_service
from modules.memory.summary_generator import SummaryGenerator
from config import get_settings

_logger = get_logger("memory")


class MemoryManager:
    """双层记忆管理器：短期原文 + 长期向量召回。"""

    def __init__(self, llm_client=None):
        self.pg = PGManager()
        self.embedding_service = get_embedding_service()
        self.summary_generator = SummaryGenerator(llm_client=llm_client)
        s = get_settings()
        self._max_history_tokens = s.max_history_tokens
        self._short_rounds = s.short_memory_rounds
        self._similarity_threshold = s.similarity_threshold

    def save_turn(
        self,
        session_id: str,
        user_id: str,
        user_query: str,
        llm_full_answer: str,
        llm_summary: str,
        summary_embedding: list,
        tool_call_log: list | None = None,
        rag_sources: list | None = None,
        query_embedding: list | None = None,
    ) -> int | None:
        """保存一轮对话全量数据（原文+摘要+摘要向量+问句向量+工具日志）。"""
        return self.pg.save_history(
            session_id, user_id,
            user_query, llm_full_answer, llm_summary,
            summary_embedding, tool_call_log, rag_sources,
            query_embedding,
        )

    def build_context(
        self,
        session_id: str,
        user_id: str,
        query: str,
        query_embedding: list | None = None,
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
        breakpoint()  # 【调试断点】记忆构建入口：可看 query / session_id / user_id
        # ---- 长期记忆：摘要向量语义召回 ----
        long_term = self._recall_long_term(query, user_id, query_embedding)

        # ---- 短期记忆：最近N轮完整对话 ----
        short_term = self.pg.get_short_term_history(
            session_id, user_id, self._short_rounds
        )

        # ---- 上下文 token 检查与压缩 ----
        context_text = self._format_context(long_term, short_term, query)
        compressed = False
        if check_token_overflow(context_text, self._max_history_tokens):
            _logger.warning("记忆上下文溢出，触发压缩（边界4）")
            long_term = self._compress_long_term(long_term, user_id, query)
            context_text = self._format_context(long_term, short_term, query)
            compressed = True

        return {
            "long_term_memory": long_term,
            "short_term_memory": short_term,
            "context_text": context_text,
            "compressed": compressed,
        }

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
        if query_embedding is None:
            query_embedding = self.embedding_service.embed(query)

        # 这行代码的作用就是：拿着刚才生成的"问题向量"，去 PostgreSQL 数据库里搜出最相关的记忆片段。
        # self.pg 你封装好的 PostgreSQL 数据库客户端对象（装了 pgvector 插件）
        # .search_memory_by_vector(...)	自定义方法，意思是"按向量搜记忆"
        # query_embedding	刚才用 embedding 模型把用户问题转成的那串数字（向量）
		# user_id	用户 ID，这里是"cli-user"
        # top_k=5	 只返回最相关的前5条
        # results	搜出来的结果，通常是列表，每条包含记忆内容+相似度分数

        # 3. 返回结果大概长这样
        # results = [
        #     {"content": "项目A后端开发已完成80%", "similarity": 0.92},
        #     {"content": "项目A前端页面还在调试", "similarity": 0.87},
        #     {"content": "下周要开项目评审会", "similarity": 0.81},
        #     {"content": "数据库迁移计划推迟了", "similarity": 0.76},
        #     {"content": "测试环境部署遇到问题", "similarity": 0.72},
        # ]

        results = self.pg.search_memory_by_vector(
            query_embedding, user_id, top_k=5
        )
        # ------------------------------------------------------------------
        # 【为什么召回后还要再滤一道（边界8）】
        #   向量搜索没有"不相关就返回空"的逻辑：只要库里还有别的记录，
        #   它就一定凑满 top_k=5 条，哪怕相似度只有 0.2。所以 pg 返回的是
        #   "最像的 5 条"，不等于"相关的 5 条"——这一步才把"最像"筛成"够像"。
        #
        # 【>= 是闭区间】恰好等于阈值的记录【保留】。想严格大于是 >，
        #   改阈值时要留意这半分开的差别：0.65 时打 0.65 分的记录会留下。
        #
        # 【顺序不变】pg 那边已按 similarity 倒序返回，过滤不重排，
        #   所以 filtered[0] 依然是最相关的一条（后面 _format_context 按此顺序拼）。
        #
        # 【空结果无需特判】results 为空（该用户没有历史）时这里直接得 []，
        #   不报错；下面那段的告警专门盯"有召回但全被滤掉"这种更可疑的情况。
        #
        # 【阈值从哪来】self._similarity_threshold ← config.py:71，
        #   默认 0.65，可用 .env 的 SIMILARITY_THRESHOLD 覆盖。调小 → 召得更宽但噪声多。
        # ------------------------------------------------------------------
        filtered = [
            r for r in results
            if r["similarity"] >= self._similarity_threshold
        ]
        if results and not filtered:
            _logger.info(
                f"历史记忆召回{len(results)}条但全部低于阈值{self._similarity_threshold}，已舍弃"
            )
        _logger.info(f"长期记忆召回: 命中={len(filtered)}/{len(results)}")
        breakpoint()  # 【调试断点】长期记忆召回结果：可看 filtered（过阈值后的记忆）
        return filtered

    def _compress_long_term(self, long_term: list, user_id: str, query: str) -> list:
        """
        上下文压缩：将多个久远摘要合并为一个全局摘要，减少 token 占用。
        """
        if len(long_term) <= 1:
            return long_term
        summaries = [item["summary"] for item in long_term]
        merged = self.summary_generator.merge_summaries(summaries)
        if not merged:
            return long_term
        # 合并后保留1条
        return [{
            "summary": merged,
            "query": query,
            "answer": merged,
            "similarity": 1.0,
            "time": "merged",
            "merged": True,
        }]

    @staticmethod
    def _format_context(long_term: list, short_term: list, query: str) -> str:
        """拼接记忆上下文为文本。"""
        blocks = []
        if long_term:
            blocks.append("【历史记忆摘要】")
            for item in long_term:
                blocks.append(f"- {item.get('summary', '')}")
        if short_term:
            blocks.append("\n【近期对话】")
            for item in short_term:
                blocks.append(f"用户: {item.get('query', '')}")
                blocks.append(f"AI: {item.get('answer', '')[:200]}")
        return "\n".join(blocks)
