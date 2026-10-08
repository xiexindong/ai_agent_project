"""
提示词组装器模块
================
把「角色 + 策略 + 上下文 + 输出格式」四层拼成最终 messages，并在拼装时做 token 预算裁剪。

【四层结构，为什么分四层】
    角色层     role          这轮对话模型"是谁"（企业助手 / 财务专员 / 审核员）
    策略层     strategy      这轮"怎么问"（zero_shot / few_shot / cot / react）
    上下文层   **context     这轮"拿什么答"（知识库、记忆、权限、工具清单）
    输出层     output_format 这轮"答成什么样"（纯文本 / JSON / Markdown）

    四层互不耦合：换策略不用动模板，换模板不用动裁剪逻辑。
    而现在 react_agent.py:305 是把模板、策略、上下文全写在一个 format() 里，
    想单独调"策略"这一层，只能在那一大段字符串里改。

【token 预算裁剪的顺序】
    上下文不是平均重要。预算不够时按优先级从低往高丢：
        权限边界 > 工具清单 > 观察结果 > 技能SOP > 知识库 > 历史记忆 > 工具历史
    权限边界排最前是因为：它是硬性约束，丢了会越权；
    历史记忆排后面是因为：它只是"可能有帮助"，丢了最多答得没那么贴人。
"""
from dataclasses import dataclass, field   # dataclass 装饰器与 field：简化组装器的配置字段声明
from typing import Dict, List, Optional   # 类型注解：字典、列表、可空

from modules.common.logger import get_logger   # 项目统一日志入口
from modules.common.token_counter import count_tokens   # token 计数器：预算裁剪与统计全靠它
from modules.common.validator import repair_json   # JSON 修复器：修模型输出的常见 JSON 毛病
from modules.prompt.builtin import get_registry   # 全局模板注册表单例：默认模板的来源
from modules.prompt.strategies import FewShotStore, build_strategy_block   # 少样本库类型与策略块生成函数
from modules.prompt.templates import PromptRegistry   # 模板注册表类型：registry 字段的类型注解

_logger = get_logger("prompt.builder")   # 本模块日志器，带模块名前缀便于按模块过滤日志

# 上下文块优先级，从高到低。不在表里的键排在最后（按名字排序，保证顺序稳定）。
_CONTEXT_PRIORITY = (   # 上下文裁剪优先级表：预算不够时按此顺序从低往高丢
    "permission_context",  # 权限边界：硬性约束，丢了会越权，优先级最高
    "tools_desc",          # 工具清单：没有它模型无从调用工具
    "observation",         # 工具观察结果：本轮刚拿到的新事实
    "skill_context",       # 技能 SOP：已激活技能时的作业流程
    "rag_context",         # 知识库参考：回答的主要依据
    "memory_context",      # 历史记忆：让回答更贴人，丢了不致命
    "tool_history",        # 历史工具调用：只在排查重复调用时有用，优先级最低
)   # 优先级元组结束

# 上下文类变量：这类变量"有就给、没有就没有"，调用方不传时填占位，不报错。
# 与之相对的是业务变量（query / answer / summaries / labels / max_iter），
# 那些缺了整个提示词就没意义，必须由模板层的 render() 抛错拦住。
# 注意这里不含 strategy_block：策略是"指令"不是"上下文"，由 build_record 直接注入，
# 不参与上下文预算竞争，否则预算一满就会先把"怎么问"丢掉，只剩一堆资料。
_CONTEXT_KEYS = set(_CONTEXT_PRIORITY)   # 上下文类变量的集合，用于判断某个键是否参与预算裁剪
_EMPTY_PLACEHOLDER = "（未提供）"            # 上下文没传时的占位文本，保证模板始终能渲染出来


# 【什么】把四层内容拼成最终 messages 的组装器
# 【作用】统一负责渲染、策略注入、token 预算裁剪，调用方只管传上下文
# 【为什么】四层互不耦合：换策略不动模板，换模板不动裁剪逻辑，避免一大段 format() 里改一处动全身
@dataclass   # 用 dataclass 自动生成 __init__，所有配置项即字段
class PromptBuilder:   # 提示词组装器类：本模块的核心，对外提供 build / build_record / system_prompt
    """提示词组装器。

    Attributes:
        template_name:    用的哪个模板（默认 qa.grounded）
        template_version: 指定版本，留空取最新
        strategy:         见 strategies.STRATEGIES
        role:             角色指令，拼在 system 开头
        output_format:    text / json / markdown
        output_schema:    json 模式下补充的字段说明
        max_context_tokens: 上下文 token 预算（不含 query 与指令本身）
        few_shot_store:   few_shot 策略的示例库
        few_shot_k:       few_shot 取几条示例
        max_iter:         react 策略的循环上限
        registry:         模板注册表，留空用全局单例
    """   # 文档字符串：逐字段说明组装器的配置项

    template_name: str = "qa.grounded"             # 用的模板名，默认知识库问答模板
    template_version: str = ""                     # 指定版本；留空取最后注册的那版
    strategy: str = "zero_shot"                    # 提问策略，见 strategies.STRATEGIES
    role: str = ""                                 # 角色指令，拼在 system 开头
    output_format: str = "text"                    # 输出格式：text / json / markdown
    output_schema: str = ""                        # json 模式补充的字段说明
    max_context_tokens: int = 3000                 # 上下文 token 预算，不含 query 与指令本身
    few_shot_store: Optional[FewShotStore] = None  # few_shot 示例库；为空则退化成 zero_shot
    few_shot_k: int = 2                            # few_shot 取几条示例
    max_iter: int = 10                             # react 策略的循环上限，写进策略块
    registry: Optional[PromptRegistry] = None      # 模板注册表；留空用全局单例

    # 残羹冷炙阈值：裁剪后剩下的 token 少于这个值就不塞了，
    # 半截上下文比没有更容易误导模型（看着像有依据，实际是被砍断的）。
    _MIN_KEEP_TOKENS: int = field(default=50, repr=False)   # 最小保留阈值：剩余预算低于 50 token 就整块丢弃；repr=False 让它不出现在对象打印里

    # ------------------------------------------------------------------
    # 【build 的返回结构】
    #   返回 OpenAI 格式的 messages：system 一条 + user 一条。
    #   system 放「角色 + 策略 + 输出格式」（这三层与具体提问无关，可复用），
    #   user 放「渲染后的模板」（随每次提问变化）。
    #   分开的好处：多轮对话时 system 不变，服务端可命中前缀缓存（省 token 也省钱）。
    # ------------------------------------------------------------------
    # 【什么】对外主入口：返回可直接喂给模型的 messages
    # 【作用】只要 messages 时用这个；要看 token 与裁剪记录用 build_record
    # 【为什么】多数调用方只需要 messages，把统计细节藏在 build_record 里让主路径保持简单
    def build(self, query: str, **context) -> List[Dict[str, str]]:   # 简化入口：返回 OpenAI 格式的 messages 列表
        rec = self.build_record(query, **context)   # 委托给完整构建方法，拿到含统计信息的完整结果
        return rec["messages"]   # 只取 messages 部分返回，丢弃统计信息

    # 【什么】完整构建：messages + token 数 + 裁剪记录
    # 【作用】排查慢查询、评测效果时用它，能看到哪块上下文被截断或丢弃
    # 【为什么】评测与排查需要完整的组装过程留痕，只给 messages 无法回答"为什么答得差"
    def build_record(self, query: str, **context) -> dict:   # 完整构建方法：组装全过程并返回详细记录
        """完整构建，除 messages 外还返回 token 数与裁剪记录（排查与评测用）。

        Returns:
            {
              "messages": [{"role": "system", ...}, {"role": "user", ...}],
              "system": str, "user": str,
              "template": "qa.grounded(v1)", "strategy": "few_shot",
              "prompt_tokens": int,       # system + user 合计
              "context_tokens": int,      # 裁剪后上下文实际占用
              "trimmed": [str, ...],      # 裁剪动作，空列表表示没超预算
            }
        """   # 文档字符串：完整列出返回字典的每个字段
        registry = self.registry or get_registry()   # 取注册表：调用方注入了就用注入的，否则用全局单例
        tpl = registry.get(self.template_name, self.template_version)   # 按模板名+版本取模板对象；版本留空取最新
        needed = tpl.variables   # 取模板声明的变量集合，后面靠它判断模板有哪些插位

        # ------------------------------------------------------------------
        # 【策略段到底进 user 还是进 system】取决于模板有没有 {strategy_block} 插位：
        #   有插位（qa.grounded）→ 策略段作为变量渲染进 user，system 里不再拼；
        #   没插位（react.reason）→ 策略段整段进 system，user 只放上下文。
        #   两条路只走一条，否则同一段指令会出现两次，白烧 token。
        #
        # 【工具清单同理】strategy=react 的策略块自带「## 可用工具」，
        #   而 react.reason 模板也有 {tools_desc} 插位，两处都填就是同一份清单两遍。
        #   判断依据同样是 needed：模板自己有插位，策略块就不带清单。
        # ------------------------------------------------------------------
        tools_desc = str(context.get("tools_desc", "") or "")   # 先取出工具清单原文备用；None 兜成空串，供后面拼 system 用
        ctx, trimmed = self._fit_context(needed, context)   # 按 token 预算裁剪上下文：返回塞进模板的变量字典 + 裁剪账单
        if "strategy_block" in needed:   # 模板有 {strategy_block} 插位 → 策略段渲染进 user
            ctx["strategy_block"] = build_strategy_block(   # 生成策略指令文本并作为变量放进上下文
                self.strategy,   # 策略名：zero_shot / few_shot / cot / react
                query=query,   # 用户问题：few_shot 检索示例要用
                store=self.few_shot_store,   # few_shot 示例库
                k=self.few_shot_k,   # few_shot 取几条示例
                tools_desc="",   # 工具清单留空：清单已由模板自己的 {tools_desc} 插位承载，避免重复
                max_iter=self.max_iter,   # react 循环上限
            )   # build_strategy_block 调用结束
        if "query" in needed:   # 模板有 {query} 插位 → 把用户问题放进上下文
            ctx["query"] = query   # query 是业务变量，不参与预算裁剪，直接放入
        user = tpl.render(**ctx)   # 渲染模板得到 user 段；缺业务变量时 render() 抛 PromptError

        system = self.system_prompt(   # 生成 system 段：角色 + 策略 + 输出格式
            query=query,   # 传 query 供 few_shot 策略检索（策略段进 system 的场景）
            tools_desc="" if "tools_desc" in needed else tools_desc,   # 模板自带工具插位时不再重复拼清单，否则把清单交给策略块
            include_strategy="strategy_block" not in needed,   # 模板有策略插位时 system 不再拼策略段，避免同一段指令出现两次
        )   # system_prompt 调用结束

        messages = []   # 初始化 messages 列表：最终要喂给模型的结构
        if system:   # system 非空才追加（role 和策略都可能为空）
            messages.append({"role": "system", "content": system})   # 追加 system 消息：角色 + 策略 + 输出格式
        messages.append({"role": "user", "content": user})   # 追加 user 消息：渲染后的模板正文

        prompt_tokens = count_tokens(system) + count_tokens(user)   # 统计整段提示词的 token 数：system + user 合计
        _logger.debug(   # debug 日志记录本次组装摘要，便于排查与性能分析
            f"提示词组装: {tpl.name}({tpl.version}) strategy={self.strategy} "   # 日志内容第一段：模板名、版本、策略
            f"tokens={prompt_tokens}"   # 日志内容第二段：总 token 数
        )   # _logger.debug 调用结束
        return {   # 组装并返回完整构建记录
            "messages": messages,   # OpenAI 格式消息列表，可直接喂给模型
            "system": system,   # system 段原文，便于单独检查
            "user": user,   # user 段原文，便于单独检查
            "template": f"{tpl.name}({tpl.version})",   # 实际使用的模板标识，如 qa.grounded(v1)
            "strategy": self.strategy,   # 实际使用的策略名
            "prompt_tokens": prompt_tokens,   # 整段提示词 token 数（system + user 合计）
            "context_tokens": sum(count_tokens(v) for v in ctx.values() if isinstance(v, str)),   # 裁剪后上下文实际占用 token 数（只统计字符串值）
            "trimmed": trimmed,   # 裁剪账单：每行一条截断/丢弃动作，空列表表示没超预算
        }   # 返回字典结束

    # ------------------------------------------------------------------
    # 【只生成 system 段】给「system 固定、user 每轮变」的场景用。
    #   react_agent 的 reason_step 正是这种：角色与流程每轮都一样，变的只是
    #   query 与上下文，所以 system 在构造时算一次缓存起来，每轮只重拼 user
    #   —— 省一次渲染，也让服务端能命中前缀缓存（prefix caching）。
    #
    # 【include_strategy=False 的时机】
    #   模板自带 {strategy_block} 插位时（qa.grounded），策略段留在 user，
    #   system 就不重复拼，见 build_record 里的分支。
    #
    # 【输入输出对照（strategy=react, max_iter=10）】
    #   输入  role="你是企业级智能AI助手..."  tools_desc=""
    #   输出  "你是企业级智能AI助手...\n\n## 工作流程（ReAct）\n1. Reason...\n\n## 可用工具\n（无可用工具）"
    # ------------------------------------------------------------------
    # 【什么】只生成 system 段（角色 + 策略 + 输出格式）
    # 【作用】system 固定、user 每轮变的场景可缓存它，还能命中前缀缓存
    # 【为什么】多轮对话中 system 不变，缓存一次既省渲染又让服务端命中前缀缓存省钱
    def system_prompt(   # system 段生成方法：拼装「角色 + 策略 + 输出格式」三层
        self,   # 实例自身
        role: str = "",   # 角色指令覆盖值；留空用实例配置 self.role
        query: str = "",   # 用户问题；few_shot 策略检索示例要用
        tools_desc: str = "",   # 工具清单文本；react 策略拼进「## 可用工具」节
        include_strategy: bool = True,   # 是否拼策略段；模板自带 {strategy_block} 插位时传 False
    ) -> str:   # 返回值：完整的 system 段文本，可能为空串
        parts = [role or self.role]   # 第一段放角色指令：参数优先，参数为空用实例配置
        if include_strategy:   # 需要拼策略段（模板没有 {strategy_block} 插位的场景）
            parts.append(   # 把策略指令文本追加为第二段
                build_strategy_block(   # 调用策略模块生成「怎么问」指令
                    self.strategy,   # 策略名
                    query=query,   # 用户问题：few_shot 检索要用
                    store=self.few_shot_store,   # few_shot 示例库
                    k=self.few_shot_k,   # few_shot 取几条示例
                    tools_desc=tools_desc,   # react 的工具清单
                    max_iter=self.max_iter,   # react 循环上限
                )   # build_strategy_block 调用结束
            )   # parts.append 调用结束
        fmt = self._format_instruction()   # 生成输出格式约束段；text 模式返回空串
        if fmt:   # 有格式约束（json / markdown）才追加
            parts.append(fmt)   # 把格式约束追加为最后一段
        return "\n\n".join(p for p in parts if p)   # 各段用空行连接；过滤空段避免产生多余空行

    # ------------------------------------------------------------------
    # 【贪心装包：按优先级顺序一块一块往预算里塞】
    #   塞得下 → 整块保留；塞不下 → 截断到剩余预算；剩余预算太少 → 整块丢弃。
    #
    # 【输入输出对照（max_context_tokens=500）】
    #   输入  rag_context=1200 token 的长文档，memory_context=80 token
    #   输出  rag_context 截断到 500 token，memory_context 丢弃
    #         trimmed = ["rag_context: 截断 1200 → 486 token",
    #                    "memory_context: 丢弃 80 token（预算已满）"]
    #
    # 【为什么不截断 query】问题是本次请求的全部意义，砍它等于换了个问题。
    # ------------------------------------------------------------------
    # 【什么】按预算装上下文：能塞就塞，塞不下就截断或丢弃
    # 【作用】保证总 token 不超 max_context_tokens（默认 3000），并记录裁剪动作
    # 【为什么】上下文不是平均重要：按优先级贪心装包，预算紧时先丢低价值块，保硬性约束
    def _fit_context(self, needed, context: dict) -> tuple[dict, List[str]]:   # 上下文裁剪方法：返回 (塞进模板的变量字典, 裁剪账单)
        ordered = [k for k in _CONTEXT_PRIORITY if k in needed and k in _CONTEXT_KEYS]   # 按优先级表顺序收集模板实际需要且属于上下文类的变量
        ordered += sorted(   # 追加不在优先级表里的上下文变量：排最后，按名字排序保证顺序稳定
            k for k in needed if k in _CONTEXT_KEYS and k not in _CONTEXT_PRIORITY   # 筛选条件：是上下文类变量但不在优先级表里
        )   # 列表拼接结束，ordered 为最终的裁剪顺序
        # 上下文变量没传 → 填占位，保证模板能渲染出来；
        # 业务变量（query/answer/...）不在这里，缺了由 tpl.render() 抛错。
        #
        # 【这一行做了什么】按 ordered 的优先级顺序，把每个上下文变量统一成
        #   字符串：调用方传了值（含数字/对象）→ str() 归一；没传或是空
        #   （None/""/0/[] 都算 falsy）→ 填占位符"（未提供）"。
        #   先归一化，后面循环只跟字符串打交道，不用反复判类型。
        filled = {k: str(context.get(k) or _EMPTY_PLACEHOLDER) for k in ordered}   # 归一化：传了值 str() 转字符串；没传或 falsy 填占位符

        # ------------------------------------------------------------------
        # 【贪心装包循环】像往一个固定大小的行李箱里按优先级装东西，
        #   budget=箱子容量（默认 3000 token），used=已占空间。
        #   每处理一块只有三种下场：整块装下 / 砍断装一截 / 整个丢弃。
        #
        #   【一次完整推演】budget=300，优先级顺序：
        #     permission(40) → tools(60) → observation(120) → rag(200) → memory(80)
        #     permission: used 0+40≤300   → 整块，used=40
        #     tools:      40+60≤300       → 整块，used=100
        #     observation:100+120≤300     → 整块，used=220
        #     rag:        220+200>300     → 剩 80 ≥ 50，截断到 80，used=300
        #     memory:     300+80>300      → 剩 0 < 50，整块丢弃，填占位
        # ------------------------------------------------------------------
        budget = self.max_context_tokens   # 箱子容量：上下文 token 总预算（默认 3000）
        used = 0   # 已占空间：初始为 0
        kept: dict = {}          # 最终塞进模板的变量；每个键都有值（含占位符），保证 render 不缺变量
        trimmed: List[str] = []  # 裁剪账单：每个被截断/丢弃的动作留一行，build_record 原样带出给排查用
        for key in ordered:   # 按优先级从高到低逐块处理
            text = filled[key]   # 取这一块归一化后的文本
            if not text:   # 空文本（理论上占位符不会走到这，防御性判断）
                continue                         # 空文本不花预算（占位符不算空，会正常参与）
            cost = count_tokens(text)            # 这一整块要占多少 token
            # 【下场①：整块装得下】已用 + 本块 ≤ 容量 → 原样保留，累加占用
            if used + cost <= budget:   # 预算还够装下整块
                kept[key] = text   # 原样保留这一块
                used += cost   # 累加已占空间
                continue   # 处理下一块
            # 走到这里说明整块装不下，看还剩多少空间
            remain = budget - used   # 计算剩余预算
            # 【下场②：剩余空间够装一截】≥50 token 才砍断，否则残段没有保留价值
            if remain >= self._MIN_KEEP_TOKENS:   # 剩余空间够装一截（不低于最小保留阈值）
                clipped = self._clip(text, remain)   # 按 token 比例截前段 + 加"（已截断）"，_clip 见 296 行
                kept[key] = clipped   # 把截断后的残段塞进模板
                trimmed.append(f"{key}: 截断 {cost} → {count_tokens(clipped)} token")  # 账单写明砍前砍后，看得到损失
                used = budget                          # 箱子从此装满，后面的块只剩"丢弃"一条路
            else:   # 剩余空间不足 50 token → 整块丢弃
                # 【下场③：剩余空间 < 50】塞残段只会让模型看到半截依据，
                #   不如明确告诉它"这一块没有"。
                # 【丢弃也要留占位符】整块不塞进 prompt 时，模板里的 {xxx} 仍然要填。
                #   不填 → render() 以"缺少变量"抛错，一次裁剪把整轮对话搞挂；
                #   填"（未提供）" → 模型明确知道这一块没有，比留空更不容易瞎编。
                kept[key] = _EMPTY_PLACEHOLDER   # 填占位符：模板渲染不缺变量，模型明确知道此块无内容
                trimmed.append(f"{key}: 丢弃 {cost} token（预算已满）")   # 账单记录丢弃动作与损失量
        return kept, trimmed   # 返回 (塞进模板的变量字典, 裁剪账单)

    # ------------------------------------------------------------------
    # 【按 token 反推该留多少字符】
    #   tiktoken 不能"只编前 N 个 token"再倒推文本（编码不可逆到字符边界），
    #   所以按「token : 字符」的比例估算，再留 12 个字符给省略标记。
    #   截断只是保底手段，真正该做的是上游就别塞这么长的上下文进来。
    # ------------------------------------------------------------------
    # 【什么】把文本按 token 上限截短
    # 【作用】超出预算时保留前段并追加「…（已截断）」标记，给模型明确提示
    # 【为什么】tiktoken 无法从 token 数反推字符边界，只能按比例估算；截断是保底手段而非正常路径
    @staticmethod   # 静态方法：截断逻辑不依赖实例状态
    def _clip(text: str, limit_tokens: int) -> str:   # 文本截短方法：按 token 上限保留前段
        total = count_tokens(text)   # 先算全文实际 token 数
        if total <= limit_tokens:   # 没超限 → 不需要截
            return text   # 原样返回
        keep = max(0, int(len(text) * (limit_tokens / total)) - 12)   # 按「token:字符」比例反推该留多少字符，再减 12 留给省略标记；max(0,...) 防负数
        return text[:keep] + "\n…（已截断）"   # 保留前段 + 追加截断标记，明确告知模型内容被砍断

    # 【什么】生成输出格式约束段
    # 【作用】json / markdown 才产出指令，text 返回空串即不附加约束
    # 【为什么】格式约束放 system 而非 user：模型对 system 的格式要求服从度更高，且多轮时不被历史淹没
    def _format_instruction(self) -> str:   # 格式约束生成方法：按 output_format 产出对应指令文本
        """输出格式约束。放 system 而不是 user：
        模型对 system 里的格式要求服从度更高，且多轮时不会随历史被淹没。"""   # 文档字符串：解释约束放 system 的原因
        if self.output_format == "json":   # JSON 输出模式
            schema = f"\n字段: {self.output_schema}" if self.output_schema else ""   # 有字段说明就补上「字段: ...」，没有就不补
            return (   # 返回 JSON 输出约束指令
                "## 输出格式\n只输出一个 JSON 对象，不要用 ``` 包裹，"   # 核心要求：裸 JSON，不要代码块包裹
                "不要写任何解释文字。" + schema   # 补充要求：不要解释文字 + 可选的字段说明
            )   # return 表达式结束
        if self.output_format == "markdown":   # Markdown 输出模式
            return "## 输出格式\n用 Markdown：一级标题写结论，下面用要点列出依据。"   # 返回 Markdown 结构要求
        return ""   # text 模式（默认）：不附加任何格式约束


# ----------------------------------------------------------------------
# 【结构化输出的解析：与 builder 配套的收尾】
#   builder 负责"让模型输出 JSON"，这里负责"把输出变成 dict"。
#   复用 validator.repair_json：它能修尾逗号、缺右括号等常见毛病
#   （repair 失败返回 None，调用方按 None 处理，不要假装解析成功）。
#
# 【输入输出对照】
#   输入 '{"intent": "报销咨询", "confidence": 0.9,}'   ← 尾逗号，json.loads 会失败
#   输出 {"intent": "报销咨询", "confidence": 0.9}
#   输入 '我觉得这是报销咨询'                            ← 完全不是 JSON
#   输出 None
# ----------------------------------------------------------------------
# 【什么】把模型输出解析成 dict
# 【作用】配合 output_format=json 收尾；解析不了返回 None 由调用方处理
# 【为什么】模型输出常带尾逗号等小毛病，直接 json.loads 会失败；解析失败必须返回 None 而非假装成功
def parse_json_output(text: str) -> Optional[dict]:   # JSON 输出解析函数：模型文本 → dict 或 None
    if not text:   # 空文本（None 或空串）→ 无可解析内容
        return None   # 直接返回 None
    return repair_json(text.strip())   # 去首尾空白后交给 repair_json 修复并解析；修不好返回 None
