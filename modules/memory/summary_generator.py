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
        # ------------------------------------------------------------------
        # 【整块在做什么】把一轮"提问 + 回答"压成一句话摘要（≤50 字）。
        #   产物有两个去处：① 落库成 chat_history.llm_summary（被向量化的那一列）
        #   ② 被 _format_context 拼进 prompt 的【历史记忆摘要】。
        #   三条分支按代价递增：不调 LLM → 不调 LLM → 调 LLM。
        #
        # 【分支1：没有回答】answer 为空（LLM 调用失败/超时/被拦截）时，
        #   退而用提问的前 50 字符当摘要 —— 内容不完整，但保证这轮对话
        #   仍入得了库、检索得到，不至于整轮凭空消失。
        if not answer:
            return query[:50]
        # 【分支2：短回答不调 LLM】count_tokens(answer) < 100 时直接截断。
        #   【为什么用 token 而非字符】LLM 按 token 计费与限长，阈值口径要和它统一
        #     （count_tokens 走 tiktoken cl100k_base，中文 1 字 ≈ 1~2 token），
        #     所以 <100 token 约等于"中文 50~100 字以内的短回答"。
        #   【replace 换行是必须的】摘要要单行：它会被拼进 _format_context 的
        #     "- {summary}" 列表项，也会落库成单行字段，换行会破坏列表格式与日志。
        #   【50 是硬截断】_SUMMARY_PROMPT 里"不超过50字"只是提示词软约束，
        #     这里 answer[:50] 才是代码层的硬保证（按字符算，含标点）。
        if count_tokens(answer) < 100:
            return answer[:50].replace("\n", " ").strip()
        # 【分支3：走 LLM 摘要】入参先截断再填模板。
        #   【数值含义】query[:500] / answer[:2000] 是【字符】上限而不是 token，
        #     目的是防止超长文本把 prompt 撑爆（同时也是成本控制）。
        prompt = self._SUMMARY_PROMPT.format(query=query[:500], answer=answer[:2000])
        summary = self._llm.invoke(prompt)
        # 【清引号】模型常把摘要用引号包起来（提示词以"精简摘要:"结尾，容易被
        #   当成对话续写），两层 strip 分别清掉英文双引号 " 与单引号 '。
        #   【局限】只覆盖英文半角引号；模型若输出中文弯引号，这里清不掉。
        summary = summary.strip().strip('"').strip("'")
        # 【日志】只记前 30 / 50 字符，避免长文本刷屏；debug 级别默认不输出。
        _logger.debug(f"摘要生成: query='{query[:30]}...' -> summary='{summary[:50]}...'")
        # ------------------------------------------------------------------
        return summary

    @handle_exception(default=None, log_msg="历史摘要合并异常")
    def merge_summaries(self, summaries: list[str]) -> str:
        """合并多个久远摘要为全局摘要（上下文压缩用）。"""
        if not summaries:
            return ""
        if len(summaries) == 1:
            return summaries[0]
        # ------------------------------------------------------------------
        # 【整块在做什么】多条历史摘要 → 一个带编号的整段字符串 → 交给 LLM 合并。
        #   注意这是【第二次 LLM 调用】：joined 是"合并摘要"那次调用的输入，
        #   不是回答用户问题的那次调用（那次在 react_agent 里另算）。
        #
        # 【第1步：list → 带序号的文本块】
        #   enumerate(summaries) 同时给出下标 i 与内容 s；f"{i+1}. {s}" 加序号；
        #   "\n".join(...) 把字符串列表拼成【一个】字符串（list → str）。
        #
        #   【输入输出对照】
        #     summaries = ["数学计算:3*5=15", "用户问天气", "讨论了部署方案"]   （3 条）
        #            ↓ enumerate → (0,"数学计算:3*5=15") (1,"用户问天气") (2,"讨论了部署方案")
        #            ↓ f"{i+1}. {s}"
        #     joined    = "1. 数学计算:3*5=15\n2. 用户问天气\n3. 讨论了部署方案"
        #
        #   【为什么 +1】enumerate 下标从 0 起，人读的编号从 1 起，不 +1 会拼出
        #     "0. xxx"；_MERGE_PROMPT 要求"保留时间顺序"，序号就是顺序的可视锚点。
        #
        #   【依赖警告】分隔符必须写 "\n"，写成 "n" 会让 3 条挤成一行，
        #     不报错、不抛异常 —— 属于静默降质（LLM 仍猜得出大意，只是质量下滑）。
        #     走到这里 summaries 至少有 2 条：56-59 行已把空/单条提前 return。
        # ------------------------------------------------------------------
        joined = "\n".join(f"{i+1}. {s}" for i, s in enumerate(summaries))
        # 【第2步：填模板】joined 替换进 _MERGE_PROMPT 的 {summaries} 占位符。
        #   【为什么不直接传 list】.format() 只接受 str，传 list 会被 str() 成
        #     "['数学计算:3*5=15', '用户问天气']" —— 带方括号和引号的 Python 字面量，
        #     既占 token 又看不出条目边界。
        #   【为什么这样安全】_MERGE_PROMPT 内除 {summaries} 外没有别的花括号，
        #     摘要正文即使含 {} 也不会被二次解析成占位符（它是"参数值"，不是模板）。
        
        #   这句话生成一个摘要
        prompt = self._MERGE_PROMPT.format(summaries=joined)
        # 【第3步：调 LLM 合并】invoke 是一次同步阻塞调用（发 HTTP 给 DeepSeek）。
        #   【软约束不是硬约束】"不超过100字"只写在提示词里，代码既不校验也不截断，
        #     实际长度由模型决定；本项目真正的硬截断只在 generate_summary 里
        #     （43 行查空、46 行截 50 字符、47 行入参截 500/2000）。
        merged = self._llm.invoke(prompt)
        # ------------------------------------------------------------------
        return merged.strip()
