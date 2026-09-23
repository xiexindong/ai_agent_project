"""
摘要生成模块
============
调用 DeepSeek LLM 为每轮对话生成精简摘要。
摘要用于向量入库 + 上下文投喂，是长记忆语义召回的核心。
"""
# 【什么】标准库与项目内依赖
# 【作用】handle_exception 统一兜异常、count_tokens 判长短、get_registry 取提示词模板
from modules.common.logger import get_logger
from modules.common.exceptions import LLMException, handle_exception
from modules.common.token_counter import count_tokens
from modules.prompt import get_registry

_logger = get_logger("summary")                     # 模块级日志器，前缀 summary


# 【什么】调用 LLM 生成 / 合并摘要的生成器
# 【作用】把长对话压成一句话，是向量入库与上下文投喂的输入源
class SummaryGenerator:
    """LLM 摘要生成器。

    【提示词从哪来】两条提示词都取自 modules/prompt 的注册表：
        summary(v1)  单轮摘要（变量 query / answer）
        merge(v1)    多轮摘要合并（变量 summaries）
    正文与迁移前的 _SUMMARY_PROMPT / _MERGE_PROMPT 逐字一致，行为不变；
    迁走之后，改措辞在 builtin.py 一处改，还能注册 v2 做 A/B 对比。
    """

    # 【什么】持有 LLM 客户端，未传就自建
    # 【作用】函数内延迟导入避开循环依赖；测试时可注入假 client 替换真实 LLM
    def __init__(self, llm_client=None):
        if llm_client is None:
            from modules.agent.llm import LLMClient
            llm_client = LLMClient()
        self._llm = llm_client                      # 实际发请求的对象，两个方法都只用它 invoke

    # 【什么】为单轮对话生成一句话摘要
    # 【作用】短答案直接截断省掉 LLM 调用，长答案才拼提示词交给 LLM 压成一句
    # ------------------------------------------------------------------
    # 【这个装饰器在干什么】handle_exception 是本项目自研的「异常统一处理」装饰器
    #   （实现在 modules/common/exceptions.py），一次干三件事：
    #     1. 捕获异常       —— 函数体里不用再写 try/except
    #     2. 记一条错误日志  —— 日志前缀由 log_msg 指定，这里是「摘要生成异常」
    #     3. 决定怎么处置    —— 由 reraise 控制：抛出去（True），还是返回 default 兜底值（False）
    #
    # 【⚠ 本行的写法有一处矛盾】default=None 和 reraise=True 同时出现
    #   看实现就明白（exceptions.py 第 82-89 行）：reraise 为 True 时，
    #   两个 except 分支的结尾都是 raise，永远走不到 return default。
    #   所以这里的 default=None 是个「死参数」，改成任何值行为都不变 ——
    #   它更像复制粘贴留下的痕迹，删掉反而更清楚。
    #
    # 【为什么这里必须 reraise=True】因为上层的「降级策略」是写在 except 里的：
    #       react_agent.py 第 473-481 行
    #           try:
    #               summary = self.memory.summary_generator.generate_summary(...)
    #               return summary
    #           except Exception as e:
    #               _logger.warning(f"摘要生成降级: {e}")
    #           return state.user_query[:50]     ← 降级为「用提问的前 50 字当摘要」
    #   假如这里改成 reraise=False、失败时返回 None，上层就不会进 except 分支，
    #   而是直接 return 这个 None —— 摘要变成 None 写进向量库，检索时永远召不回
    #   这段记忆，且全程没有任何报错。这正是「静默失败」。
    #   所以 reraise=True 不是随手选的：它是「上层降级逻辑能否被触发」的前提。
    #
    # 【它所在的调用链】注意有两层同名函数，别搞混：
    #       graph.py:174         ReactAgent.generate_summary(state)    ← 参数是 AgentState
    #             ↓
    #       react_agent.py:475   SummaryGenerator.generate_summary(query, answer)  ← 本行
    #             ↓
    #       llm.py               self._llm.invoke(prompt)  ← LLMClient 自己还有一层 @retry
    #   三者分工：装饰器负责「记日志」，上层负责「降级」，互不重叠。
    #
    # 【对比同文件第 51 行的 merge_summaries()：两种错误处理风格并存】
    #   generate_summary → reraise=True，失败靠「抛异常」告诉上层，上层用 try/except 接
    #   merge_summaries  → reraise=False（默认），失败靠「返回值」告诉上层，
    #                      上层 memory_manager.py 第 194 行用 `if not merged:` 检查
    #   两种都能工作，但在同一个文件里混用两套风格，读代码时必须逐个确认是哪一种，
    #   属于「认知成本」而非「功能缺陷」。若要统一，通常按关键 / 非关键路径分：
    #       关键路径（数据要入库）→ 抛异常，强制失败被看见
    #       非关键路径（可退化）  → 返回兜底值，由调用方自行判断
    #
    # 【察觉不足】两个 except 分支的 reraise 行为其实并不一致（exceptions.py 第 78-89 行）
    #       except AgentException → raise                     原样抛出，保留原类型与 message/detail
    #       except Exception      → raise AgentException(log_msg, str(e))
    #                                                         被「改装」成 AgentException，原始类型丢失
    #   后果：上层若想写 `except LLMException` 精准捕获 LLM 报错，
    #         碰到「非 AgentException 的未知异常」时拿到的是 AgentException，捕不到。
    # ------------------------------------------------------------------
    @handle_exception(default=None, log_msg="摘要生成异常", reraise=True)
    def generate_summary(self, query: str, answer: str) -> str:
        """为一轮对话生成精简摘要。"""
        if not answer:
            return query[:50]                       # 没有回答时，用提问的前 50 字当摘要
        if count_tokens(answer) < 100:              # 100 token 以内算短答案，不再调 LLM
            return answer[:50].replace("\n", " ").strip()    # 取前 50 字，去掉换行与首尾空白
        prompt = get_registry().render(
            "summary", query=query[:500], answer=answer[:2000]   # 提问截 500 字、回答截 2000 字再入模板
        )
        summary = self._llm.invoke(prompt)
        #"***Hello***" text = "   Hello World!   " text = "\n\t Hello World! \n" ==> hello
        summary = summary.strip().strip('"').strip("'")
        _logger.debug(f"摘要生成: query='{query[:30]}...' -> summary='{summary[:50]}...'")
        return summary

    # 【什么】把多条久远摘要合并成一条全局摘要
    # 【作用】上下文超阈值时的压缩手段，一条换多条以省 token；失败返回 None 由上层判断
    @handle_exception(default=None, log_msg="历史摘要合并异常")
    def merge_summaries(self, summaries: list[str]) -> str:
        """合并多个久远摘要为全局摘要（上下文压缩用）。"""
        if not summaries:
            return ""
        if len(summaries) == 1:
            return summaries[0]
        # 给每条摘要加上「序号. 」前缀，再用换行拼成一整段，填进 merge 模板的 {summaries}。
        #
        #   处理前  summaries = ['用户问了 3*5 等于多少', '用户问了今天天气', '用户问了股票走势']
        #   处理后  joined    = '1. 用户问了 3*5 等于多少\n2. 用户问了今天天气\n3. 用户问了股票走势'
        #
        #   渲染进 prompt 实际长这样（模板见 modules/prompt/builtin.py 的 merge）：
        #       历史摘要列表:
        #       1. 用户问了 3*5 等于多少
        #       2. 用户问了今天天气
        #       3. 用户问了股票走势
        #
        #       合并摘要:
        #
        # 拆开看三步（enumerate 从 0 起，所以 i+1 才是给人看的 1、2、3）：
        #   enumerate(summaries) → (0, '用户问了 3*5...'), (1, '用户问了今天天气'), (2, ...)
        #   f"{i+1}. {s}"        → '1. 用户问了 3*5...',    '2. 用户问了今天天气',    '3. ...'
        #   "\n".join(...)       → 用换行把上面几项连成一个字符串
        # 加编号是因为 merge 模板要求「保留时间顺序」，编号能让 LLM 分清先后。
        #
        # 边界：本行只处理 2 条及以上 —— 空列表在第 105-106 行 return ""，
        #       单条在第 107-108 行 return summaries[0]，都走不到这里。
        joined: str = "\n".join(f"{i+1}. {s}" for i, s in enumerate(summaries))
        
        prompt = get_registry().render("merge", summaries=joined)
        merged = self._llm.invoke(prompt)
        return merged.strip()
