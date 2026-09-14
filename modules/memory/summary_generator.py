"""
摘要生成模块
============
调用 DeepSeek LLM 为每轮对话生成精简摘要。
摘要用于向量入库 + 上下文投喂，是长记忆语义召回的核心。
"""
from modules.common.logger import get_logger
from modules.common.exceptions import LLMException, handle_exception
from modules.common.token_counter import count_tokens

_logger = get_logger("summary")


class SummaryGenerator:
    """LLM 摘要生成器。"""

    _SUMMARY_PROMPT = (
        "请将以下用户提问与AI回答，提炼为一句话精简摘要（不超过50字）。"
        "摘要需保留核心信息（问题主题+关键结论），用于后续语义检索匹配。\n\n"
        "用户提问: {query}\n"
        "AI回答: {answer}\n\n"
        "精简摘要:"
    )

    _MERGE_PROMPT = (
        "请将以下多轮对话摘要合并为一个全局摘要（不超过100字），"
        "保留时间顺序与关键信息，去除冗余细节，用于压缩上下文历史。\n\n"
        "历史摘要列表:\n{summaries}\n\n"
        "合并摘要:"
    )

    def __init__(self, llm_client=None):
        # 延迟导入 LLMClient 避免循环依赖
        if llm_client is None:
            from modules.agent.llm import LLMClient
            llm_client = LLMClient()
        self._llm = llm_client

    @handle_exception(default=None, log_msg="摘要生成异常", reraise=True)
    def generate_summary(self, query: str, answer: str) -> str:
        """为一轮对话生成精简摘要。"""
        if not answer:
            return query[:50]
        # 短回答直接截断作为摘要，省 token
        if count_tokens(answer) < 100:
            return answer[:50].replace("\n", " ").strip()
        prompt = self._SUMMARY_PROMPT.format(query=query[:500], answer=answer[:2000])
        summary = self._llm.invoke(prompt)
        summary = summary.strip().strip('"').strip("'")
        _logger.debug(f"摘要生成: query='{query[:30]}...' -> summary='{summary[:50]}...'")
        return summary

    @handle_exception(default=None, log_msg="历史摘要合并异常")
    def merge_summaries(self, summaries: list[str]) -> str:
        """合并多个久远摘要为全局摘要（上下文压缩用）。"""
        if not summaries:
            return ""
        if len(summaries) == 1:
            return summaries[0]
        joined = "\n".join(f"{i+1}. {s}" for i, s in enumerate(summaries))
        prompt = self._MERGE_PROMPT.format(summaries=joined)
        merged = self._llm.invoke(prompt)
        return merged.strip()
