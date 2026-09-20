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
        long_term = self._recall_long_term(query, user_id, query_embedding)

        short_term = self.pg.get_short_term_history(
            session_id, user_id, self._short_rounds
        )

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



        results = self.pg.search_memory_by_vector(
            query_embedding, user_id, top_k=5
        )
        filtered = [
            r for r in results
            if r["similarity"] >= self._similarity_threshold
        ]
        if results and not filtered:
            _logger.info(
                f"历史记忆召回{len(results)}条但全部低于阈值{self._similarity_threshold}，已舍弃"
            )
        _logger.info(f"长期记忆召回: 命中={len(filtered)}/{len(results)}")
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
