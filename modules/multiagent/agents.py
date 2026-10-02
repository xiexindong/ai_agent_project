"""
多 Agent 实现模块
==================
三类 Agent：Supervisor（规划 + 汇总）、Worker（执行子任务）、Reviewer（终审）。

【编排模式为什么选 Supervisor-Worker，而不是"群聊式多 Agent"】
    群聊式（所有人广播、谁想接谁接）在演示里好看，但有两个工程问题：
      1. 收敛不可控：没人负责收口，可能出现两个 Agent 重复干同一件事
      2. 成本不可控：每条消息广播给所有人，token 随人数平方增长
    Supervisor-Worker 是中心化的：一个大脑拆任务、派活、收口，
    执行者只见自己的指令，既省 token 又好排查（trace 里有明确的派单人）。

【单次执行的三种形态】
    A 直接答   —— 无工具，一次 invoke 出结论
    B 带工具   —— 最多 spec.max_steps 轮的 Function Calling 循环
    C 带图     —— 先让 Vision 模型把图片"翻译"成文字，再走 A/B
                  （为什么不把图直接塞给每个 Agent：图 token 很贵，
                    且多数执行者只需要图里的结论，不需要原图）
"""
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from pydantic import BaseModel, Field

from modules.common.logger import get_logger
from modules.multiagent.models import (
    AgentOutput,
    AgentRole,
    AgentSpec,
    PlanResult,
    SubTask,
    json_schema_hint,
)
from modules.multimodal.media import MultiModalInput
from modules.multiagent.coordination import MessageBus
from modules.multiagent.delegation import decide_delegation

_logger = get_logger("multiagent")


# 【什么】Agent 依赖的 LLM 能力契约
# 【作用】用 Protocol 描述所需方法，测试时注入 MockLLM 即可不碰真实网络
class LLMProtocol(Protocol):
    """LLM 能力契约。真实实现是 LLMClient，测试/演示可换成 MockLLM。"""

    # 【什么】纯文本对话调用
    # 【作用】给不带工具的直接问答与汇总使用
    def invoke(self, prompt: str, system_prompt: str = "", temperature: float = 0.3,
               max_tokens: int = 2048) -> str: ...

    # 【什么】结构化输出调用（返回 dict）
    # 【作用】给规划、评审这类需要固定 JSON 字段的场景使用
    def invoke_structured(self, prompt: str, system_prompt: str = "",
                          temperature: float = 0.2) -> dict: ...

    # 【什么】带 Function Calling 的调用
    # 【作用】返回 content 与 tool_calls，供工具执行循环消费
    def invoke_with_tools(self, messages: list, tools: list, temperature: float = 0.3) -> dict: ...


# 【什么】角色 → 默认提示词的映射表
# 【作用】AgentSpec.system_prompt 留空时按角色取这里的值，团队配置不用每次都写提示词
DEFAULT_ROLE_PROMPTS: dict[AgentRole, str] = {
    AgentRole.SUPERVISOR: (
        "你是团队主管。职责：把用户问题拆成互不重复的子任务，分派给合适的成员，"
        "最后把成员的产出汇成一份完整答案。\n"
        "规则：\n"
        "1. 简单寒暄或单步可答的问题，直接给 direct_answer，不要拆任务\n"
        "2. 子任务之间能并行的就并行，有先后依赖的用 depends_on 声明\n"
        "3. 每个子任务只派给一个人，指令要具体到能直接执行"
    ),
    AgentRole.RESEARCHER: (
        "你是资料研究员。职责：基于给定的上下文和工具结果，找出与问题相关的事实。\n"
        "规则：只陈述有依据的内容；找不到就明说'未找到'，不要编造。"
    ),
    AgentRole.ANALYST: (
        "你是数据分析师。职责：对已有数据做计算、对比、归因。\n"
        "规则：给出计算过程的关键数字；结论要能被上游数据支撑。"
    ),
    AgentRole.REVIEWER: (
        "你是质量审核员。职责：检查草稿答案是否回答了用户的问题、是否有事实冲突、是否有遗漏。\n"
        "规则：只输出评审结论的结构化 JSON，不要重写答案。"
    ),
    AgentRole.GENERAL: (
        "你是通用助手。用简洁准确的语言回答用户问题。"
    ),
}


# 【什么】一次运行中所有 Agent 共享的上下文
# 【作用】把原始问题、多模态输入、上游产出、共享设施一次打包透传给每个 Agent
@dataclass
class RunContext:
    """一次运行中所有 Agent 共享的上下文（只读）。"""

    query: str = ""                                              # 用户的原始提问
    mm: Optional[MultiModalInput] = None                         # 多模态输入；无图片时为 None
    upstream: dict[str, AgentOutput] = field(default_factory=dict)  # {上游 task_id: 产出}，给下游当参考
    session_id: str = ""                                         # 会话 ID，透传给工具做隔离
    principal: Any = None                                        # 调用者身份；做权限过滤时使用
    # ------------------------------------------------------------------
    # 【为什么 bus 和 gate 挂在 ctx 上而不是塞进每个方法的参数】
    #   这两个是"全队共享的设施"，不是一个任务的入参。
    #   挂 ctx 上，Agent 内部任何位置都能拿到，
    #   方法签名不用层层透传（run → _build_prompt → … 会越来越长）。
    # ------------------------------------------------------------------
    bus: Optional[MessageBus] = None     # 实际挂的是 OrchestratedBus 门面：读消息直转真 bus，仅发消息走编排治理
    gate: Optional[Any] = None           # 人工闸门：敏感动作前拍板用
    # ------------------------------------------------------------------
    # 【委派三件套】Agent 把手里的活转给别人时用（详见 delegation.py）
    #   delegation       委派执行器；为 None = 本次运行不支持委派（默认）
    #   delegation_depth 当前处于第几层委派（主管派的活是 0，转一次 +1）
    #   delegation_chain 委派链（含所有上游委派方），防 A→B→A 死循环
    #   delegated_by     本次任务是受谁委派而来；空 = 主管直接派的
    # ------------------------------------------------------------------
    delegation: Optional[Any] = None
    delegation_depth: int = 0
    delegation_chain: list = field(default_factory=list)
    delegated_by: str = ""


# 【什么】审核者的评审结论（结构化）
# 【作用】把模型的评审输出收敛成可校验字段，避免拿到自由文本无法判定通过与否
class ReviewVerdict(BaseModel):
    """Reviewer 的评审结论（pydantic 校验，避免模型返回自由文本）。"""

    passed: bool = Field(description="草稿是否通过")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="0-1 质量分")
    issues: list[str] = Field(default_factory=list, description="发现的问题列表")
    suggestion: str = Field(default="", description="改进建议")


# 【什么】Agent 基类
# 【作用】封装「拼提示词 → 跑 LLM/工具 → 产出 AgentOutput」的通用流程
class BaseAgent:
    """Agent 基类：持有 spec、LLM、工具注册器，负责一次执行的通用流程。"""

    # 【什么】装配 Agent 的三件套：角色定义、LLM 实现、工具注册器
    # 【作用】LLM 与工具均可延迟注入；gate 是可选的人工闸门，不传即关闭 HITL
    def __init__(self, spec: AgentSpec, llm: Optional[LLMProtocol] = None,
                 tool_registry=None, gate=None):
        self.spec = spec                 # 角色定义：名称、角色、工具白名单、步数预算
        self._llm = llm                  # 注入的 LLM 实现，为空时首次用的时候再创建
        self.tools = tool_registry       # 工具注册器，为空表示该 Agent 无工具可用
        # ------------------------------------------------------------------
        # 【gate 是什么】人工闸门（HumanGate）。不为 None 时，
        #   每次工具调用前先过一道人工批准（见 _run_with_tools）。
        #   默认 None = 不启用 HITL，行为与改造前完全一致。
        # ------------------------------------------------------------------
        self.gate = gate
        self.system_prompt = spec.system_prompt or DEFAULT_ROLE_PROMPTS.get(spec.role, "")  # 自定义优先，否则取角色默认

    # 【什么】Agent 的唯一名称
    # 【作用】作为派活、trace、结果归属的统一标识，取自 spec.name
    @property
    def name(self) -> str:
        return self.spec.name

    # 【什么】惰性获取 LLM 客户端
    # 【作用】优先用注入的实现（可能是 Mock），没有才真正创建 LLMClient
    @property
    def llm(self) -> LLMProtocol:
        """延迟创建真实 LLM 客户端（没注入 Mock 时才 new LLMClient）。"""
        if self._llm is None:
            from modules.agent.llm import LLMClient
            self._llm = LLMClient()
        return self._llm

    # ----------------------------------------------------------------------
    # 多模态：把图片"翻译"成文字，再交给文本链路
    # ----------------------------------------------------------------------
    # 【什么】把图片翻译成客观文字描述
    # 【作用】让下游只承担一次视觉开销；无图或本 Agent 不开图能力时返回空串
    def describe_images(self, ctx: RunContext) -> str:
        """
        看图并给出客观描述。没有图片或本 Agent 不开图能力时返回空串。

        【为什么要"翻译"这一步】
            图片 token 成本是文本的几十倍，且不便于在 Agent 之间传递。
            让"能看图的 Agent"先看一次，把结论写成文字传给下游，
            整条链路就只有一次视觉开销。
        """
        if ctx.mm is None or ctx.mm.image_count == 0 or not self.spec.multimodal:
            return ""
        try:
            from modules.multimodal.vision import get_vision_client
            return get_vision_client().invoke(
                MultiModalInput(
                    instruction="请客观描述图片内容，列出图中可读的关键数字与文字，不要臆测。",
                    parts=ctx.mm.parts,
                )
            )
        except Exception as e:
            _logger.warning(f"[{self.name}] 图片解析失败（按无图继续）: {e}")
            return ""

    # 【什么】取本 Agent 可用工具的 Function Calling schema
    # 【作用】按 spec.tools 白名单过滤；取不到时返回空列表，退化为无工具执行
    def _tool_schemas(self) -> list:
        """按 spec.tools 白名单取工具的 Function Calling schema。"""
        if not self.tools or not self.spec.tools:
            return []
        try:
            return self.tools.get_openai_tools_schema(user_level="advanced", allowed=self.spec.tools)
        except Exception as e:
            _logger.warning(f"[{self.name}] 工具 schema 生成失败（按无工具执行）: {e}")
            return []

    # 【什么】Function Calling 循环执行器
    # 【作用】按步数预算反复「调工具 → 观察结果」，拿到不带工具调用的回答即收
    def _run_with_tools(self, prompt: str, steps_budget: int, session_id: str = "") -> tuple[str, int, list]:
        """
        Function Calling 循环：最多跑 steps_budget 轮，拿到不带工具调用的回答就收。

        【返回】(最终文本, 实际步数, 工具调用日志)
        【为什么限制轮数】不限制的话，模型可能在"调工具→看结果→再调工具"里打转，
            这和单 Agent ReAct 的 max_iterations=10 是同一个问题的不同尺度。
        """
        # ------------------------------------------------------------------
        # 【Subagent 上下文隔离的命门：messages 每次就地新建，绝不继承主对话】
        #   【为什么要这个？】主 Agent 的 messages 随轮次只增不减，迟早撞
        #       token 上限，旧信息混杂还会诱发幻觉。若让子 Agent 共享这份
        #       历史，等于把主对话的膨胀复制 N 份——"走子 Agent 省上下文"
        #       就成了一句空话，隔离必须做在数据结构层而不是提示词约定层。
        #   【对比例子】
        #       ✗ 把主 Agent 的整段 messages 透传给子 Agent：
        #         第 50 轮时每个子任务都背着几万 token 的闲聊与工具结果，
        #         而且 worker 能看到与自己任务无关的隐私内容。
        #       ✓ 每次执行只放两条：system（角色提示词）+ user（本次任务
        #         所需的最小信息包，由 _build_prompt 拼好）。工具循环产生
        #         的中间轮次也只活在这个局部变量里，方法返回即随栈帧释放。
        #   【本项目说明】这同时划定了"回流边界"：子 Agent 执行期的全部
        #       消息不跨方法边界外溢，外界唯一拿得到的是返回值
        #       AgentOutput.content（见下方 run() 的 return）。
        #       JS 类比：等价于 new Worker() 后主线程 postMessage(子任务)，
        #       Worker 在独立作用域里跑，主作用域变量它一个都看不到，
        #       最后只能 postMessage(最终结果) 回来，中间状态一律不带回。
        # ------------------------------------------------------------------
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        schemas = self._tool_schemas()
        logs: list = []
        last_text = ""
        step = 0
        while step < steps_budget:
            step += 1
            res = self.llm.invoke_with_tools(messages, schemas, temperature=self.spec.temperature)
            last_text = res.get("content") or ""
            calls = res.get("tool_calls") or []
            if not calls:
                return last_text, step, logs
            for call in calls[:1]:  # 一次只处理一个调用，避免并行工具结果错序
                name, args = call.get("name", ""), call.get("arguments", {}) or {}
                # ------------------------------------------------------------------
                # 【工具调用前的人工批准（HITL）】
                #   敏感工具（删改、外发、查他人数据）在这一步停下来等人。
                #   三种结果：
                #     放行   → 照原参数执行，或按人改过的参数执行（modify）
                #     拒绝   → 不执行，把"未获批准"作为观察结果回喂模型，
                #              让它改用其它方式或直接给出结论（而不是卡住）
                #     终止   → 向上抛异常，由 harness 收口
                #   gate 为 None 时这段整块跳过，零开销。
                # ------------------------------------------------------------------
                if self.gate is not None:
                    approved, args = self.gate.ask_tool(
                        name, args if isinstance(args, dict) else {},
                        agent=self.name, session_id=session_id,
                    )
                    if not approved:
                        logs.append({
                            "name": name, "arguments": args, "success": False,
                            "result": "", "error": "人工未批准",
                        })
                        messages.append({
                            "role": "user",
                            "content": (
                                f"工具 {name} 未获人工批准，本次调用已跳过。"
                                "请改用其它可用方式完成任务，或直接说明无法完成。"
                            ),
                        })
                        continue
                result = self.tools.execute_tool(name, args, session_id=session_id) if self.tools else None
                ok = bool(result and result.success)
                logs.append({
                    "name": name,
                    "arguments": args,
                    "success": ok,
                    "result": str(getattr(result, "result", ""))[:300],
                    "error": getattr(result, "error", "") if not ok else "",
                })
                observe = (
                    f"工具 {name} 返回：{str(getattr(result, 'result', ''))[:1000]}"
                    if ok else f"工具 {name} 失败：{getattr(result, 'error', '未知错误')}"
                )
                messages.append({"role": "user", "content": observe})
        return last_text or "已达到本 Agent 的推理步数上限，以下是已有信息下的结论。", step, logs

    # 【什么】执行一次子任务
    # 【作用】拼好提示词后按有无工具走两条路径，统一产出 AgentOutput（失败也不外抛）
    def run(self, task: Optional[SubTask], ctx: RunContext) -> AgentOutput:
        """执行一次子任务（Supervisor 之外的所有 Agent 走这里）。"""
        started = time.time()
        instruction = task.instruction if task else ctx.query
        upstream_text = self._upstream_text(ctx)
        vision_text = self.describe_images(ctx)
        # 同伴消息（横向对话）：取 bus 上发给自己的点对点消息与广播（*）,
        #   排除自己发出的，按时间正序取最近 20 条，每条正文截断 400 字符，
        #   拼成 "- [kind] sender: content"；无人发消息时返回 ""，可直接拼进 prompt
        peer_text = ctx.bus.inbox_text(self.name) if ctx.bus is not None else ""

        # ------------------------------------------------------------------
        # 【委派：先把"更适合别人的那部分"转出去，再带着结果做自己的部分】
        #   三个前置条件缺一不可：运行装配了委派器 / 本 Agent 有委派资格 / 有具体任务
        #   （没有 task 说明是主管直接问答，谈不上转派）
        #   开关在第一位：未启用时连"要不要转派"这次决策调用都不发生，
        #   保证关闭委派 = 零额外开销、行为与改造前完全一致。
        #   【转不出去会怎样】decide 解析失败、或治理不放行 → 返回 None，
        #   本 Agent 照常自己干，绝不因为"想转没转成"就让任务失败。
        # ------------------------------------------------------------------
        delegated: list = []
        dex = getattr(ctx, "delegation", None)
        if dex is not None and getattr(dex, "enabled", False) \
                and self.spec.can_delegate and task is not None:
            decision = decide_delegation(
                self.llm, instruction, dex.candidates(self.name), self.system_prompt
            )
            if decision is not None and decision.should_delegate:
                rec = dex.delegate(
                    from_agent=self.name,
                    to_agent=decision.to_agent,
                    instruction=decision.instruction,
                    parent_task_id=task.id,
                    depth=getattr(ctx, "delegation_depth", 0),
                    reason=decision.reason,
                    chain=list(getattr(ctx, "delegation_chain", []) or []),
                )
                if rec is not None:
                    delegated.append(rec)
                    _logger.info(f"[{self.name}] 已把部分工作委派给 {rec.to_agent}: {rec.instruction[:60]}")

        delegation_text = self._delegation_text(delegated)
        prompt = self._build_prompt(instruction, upstream_text, vision_text, peer_text, delegation_text)

        # ------------------------------------------------------------------
        # 【上下文隔离的最后一道边界】从这里往下，子 Agent 能看到的全部信息
        #   都已压缩进 prompt 这一个字符串；执行结束后只有 AgentOutput 回流，
        #   内部 messages、工具日志（logs 仅做账单留痕）都不进入主管的上下文。
        # ------------------------------------------------------------------
        try:
            schemas = self._tool_schemas()
            if schemas:
                content, steps, logs = self._run_with_tools(prompt, self.spec.max_steps, ctx.session_id)
            else:
                content = self.llm.invoke(prompt, system_prompt=self.system_prompt,
                                          temperature=self.spec.temperature)
                steps, logs = 1, []
            return AgentOutput(
                agent=self.name,
                role=self.spec.role,
                task_id=task.id if task else "",
                content=(content or "").strip(),
                success=True,
                steps=steps,
                tool_calls=logs,
                duration_ms=int((time.time() - started) * 1000),
                multimodal=bool(vision_text),
                delegations=delegated,                                   # 我转出去的活（账）
                delegated_by=getattr(ctx, "delegated_by", "") or "",     # 我是受谁委派来的
            )
        except Exception as e:
            _logger.error(f"[{self.name}] 执行失败: {type(e).__name__}: {e}")
            return AgentOutput.failed(
                self.name, task.id if task else "", f"{type(e).__name__}: {e}",
                role=self.spec.role,
                duration_ms=int((time.time() - started) * 1000),
                delegations=delegated,
                delegated_by=getattr(ctx, "delegated_by", "") or "",
            )

    # 【什么】把委派产出拼成可读文本
    # 【作用】给委派方当"上游输入"：成功的拿结论，失败的如实告知（模型据此改路子）
    @staticmethod
    def _delegation_text(records: list) -> str:
        """把委派回传的产出拼成 prompt 段落；没有委派时返回空串。"""
        if not records:
            return ""
        blocks = []
        for r in records:
            body = r.content if getattr(r, "success", False) else f"（委派失败）{getattr(r, 'error', '')}"
            blocks.append(f"【委派产出 · {r.to_agent}】{body}")
        return "\n\n".join(blocks)

    # 【什么】把上游产出拼成可读文本
    # 【作用】给下游提供依赖产出；每条截断到 1200 字防止上下文被单个产出占满
    def _upstream_text(self, ctx: RunContext) -> str:
        """把上游产出拼成可读文本（每条截断 1200 字，防上下文爆炸）。"""
        if not ctx.upstream:
            return ""
        blocks = [
            f"【上游产出 · {k}】\n{(o.content or o.error)[:1200]}"   # 成功取 content，失败取 error
            for k, o in ctx.upstream.items()
        ]
        return "\n\n".join(blocks)

    # 【什么】组装本 Agent 收到的最终提示词
    # 【作用】把任务、图片解析、上游产出、同伴消息按顺序拼成可执行的一段话
    def _build_prompt(self, instruction: str, upstream_text: str, vision_text: str,
                      peer_text: str = "", delegation_text: str = "") -> str:
        parts = [f"## 你的任务\n{instruction}"]
        if vision_text:
            parts.append(f"## 图片解析结果\n{vision_text}")
        if upstream_text:
            parts.append(f"## 上游产出\n{upstream_text}")
        # 【委派产出紧跟上游产出】它本质上就是"你转出去的那部分活回来了什么"，
        #   和上游产出同级，放在同伴消息之前，避免被闲聊冲淡
        if delegation_text:
            parts.append(f"## 你委派出去的工作产出\n{delegation_text}")
        # 【同伴消息放在要求之前】
        #   顺序上它是"执行时的参考信息"，和上游产出同级，
        #   放在最末尾的要求之前，避免把"直接给结论"这条指令冲淡。
        if peer_text:
            parts.append(f"## 同伴消息（同一批次内的其他成员）\n{peer_text}")
        parts.append("## 要求\n直接给出你的产出内容，不要复述任务，不要写客套话。")
        return "\n\n".join(parts)


# 【什么】主管 Agent
# 【作用】负责把用户问题拆成子任务，并在最后把成员产出汇成一份答案
class SupervisorAgent(BaseAgent):
    """主管：负责拆任务（plan）与汇总结论（synthesize）。"""

    PLAN_PROMPT = (   # 规划提示词模板：占位符为 query / modality / roster / schema
        "## 用户问题\n{query}\n\n"
        "## 输入形态\n{modality}\n\n"
        "## 可用成员\n{roster}\n\n"
        "## 成员产出规则\n- 能并行的任务不要串行（不要写无谓的 depends_on）\n"
        "- 一个任务只派一个人\n- 简单问题直接给 direct_answer\n\n"
        "## 输出结构（严格遵守）\n{schema}\n\n"
        "只输出一个 JSON 对象，不要多余文字。"
    )

    SYNTH_PROMPT = (   # 汇总提示词模板：占位符为 query / outputs
        "## 用户问题\n{query}\n\n"
        "## 团队成员产出\n{outputs}\n\n"
        "## 要求\n"
        "- 综合所有产出，给出一份完整、可直接交给用户的答案\n"
        "- 有冲突时以有数据支撑的为准，并说明分歧\n"
        "- 成员失败的部分如实说明'该部分未能完成'\n"
        "- 末尾附【摘要】一行总结"
    )

    # 【什么】把用户问题规划成 PlanResult
    # 【作用】结构化拿到任务清单；解析失败带错误信息重试一次，仍失败则降级为单任务直答
    def plan(self, ctx: RunContext, roster: list[AgentSpec]) -> PlanResult:
        """
        把用户问题拆成子任务。

        【为什么用结构化输出而不是解析自然语言】
            模型可能返回 "1. 先查… 2. 再算…" 这种列表文本，
            正则去抠 assignee / depends_on 极其脆弱，换个措辞就崩。
            喂 PlanResult 的 schema 后，模型只在格子里填空，
            model_validate 当场校验出缺字段，错了还能带错误信息重试一次。
        """
        roster_text = "\n".join(
            f"- {a.name}（{a.role.value}）: {a.description or '通用执行者'}"
            f"{'，可用工具: ' + ','.join(a.tools) if a.tools else ''}"
            for a in roster
        ) or "（无可用成员）"
        modality = (
            f"文本 + {ctx.mm.image_count} 张图片" if ctx.mm and ctx.mm.image_count
            else "纯文本"
        )
        prompt = self.PLAN_PROMPT.format(
            query=ctx.query,
            modality=modality,
            roster=roster_text,
            schema=json_schema_hint(PlanResult),
        )
        try:
            data = self.llm.invoke_structured(prompt, system_prompt=self.system_prompt)
            return PlanResult.model_validate(data)
        except Exception as first_err:
            # ------------------------------------------------------------------
            # 【为什么要带错误信息重试一次】
            #   模型最常见的错不是乱写，而是漏字段（如 subtasks 里少 assignee）。
            #   把校验错误原文回喂，它通常一次就补齐；不重试的话
            #   只能整单降级成"单 Agent 硬答"，白白浪费团队协作能力。
            # ------------------------------------------------------------------
            _logger.warning(f"[supervisor] 规划解析失败，带错误重试: {first_err}")
            try:
                retry_prompt = (
                    f"{prompt}\n\n上一次输出不合法，错误是：{first_err}\n"
                    "请修正后重新输出，务必包含所有必需字段。"
                )
                data = self.llm.invoke_structured(retry_prompt, system_prompt=self.system_prompt)
                return PlanResult.model_validate(data)
            except Exception as second_err:
                _logger.error(f"[supervisor] 规划二次失败，降级为单 Agent 直答: {second_err}")
                fallback = roster[0].name if roster else self.name
                return PlanResult(
                    thought="规划失败，降级为单任务执行",
                    subtasks=[SubTask(id="t1", assignee=fallback, instruction=ctx.query)],
                    needs_review=False,
                )

    # 【什么】把成员产出汇成最终答案
    # 【作用】成功失败的产出都写进提示词，失败部分如实标注"该部分未能完成"
    def synthesize(self, ctx: RunContext, outputs: list[AgentOutput]) -> str:
        """汇总结论。任一产出为空都如实标注，不替成员编内容。"""
        blocks = []
        for o in outputs:
            status = "成功" if o.success else "失败"
            body = o.content if o.success else f"（失败）{o.error}"
            blocks.append(f"【{o.agent} · {status}】{body}")
        prompt = self.SYNTH_PROMPT.format(query=ctx.query, outputs="\n\n".join(blocks))
        try:
            return self.llm.invoke(prompt, system_prompt=self.system_prompt)
        except Exception as e:
            _logger.error(f"[supervisor] 汇总失败，回退拼接: {e}")
            return "\n\n".join(o.content for o in outputs if o.success)


# 【什么】通用执行者 Agent
# 【作用】完全复用 BaseAgent.run，按 spec 配的工具与步数预算产出一段结论
class WorkerAgent(BaseAgent):
    """执行者：接一个子任务，可用工具，产出一段结论。"""


# 【什么】审核者 Agent
# 【作用】对汇总草稿做质量把关，输出结构化的评审结论与分数
class ReviewerAgent(BaseAgent):
    """审核者：对汇总草稿打分，输出结构化结论。"""

    REVIEW_PROMPT = (   # 评审提示词模板：占位符为 query / draft / schema
        "## 用户问题\n{query}\n\n"
        "## 待审草稿\n{draft}\n\n"
        "## 输出结构（严格遵守）\n{schema}\n\n"
        "只输出一个 JSON 对象。"
    )

    # 【什么】评审汇总草稿
    # 【作用】结构化拿到「是否通过 + 分数 + 问题」；解析失败也不阻断，按放行处理
    def review(self, ctx: RunContext, draft: str) -> AgentOutput:
        """评审草稿。解析失败不阻断主流程：返回 passed=True 的中性结论。"""
        started = time.time()
        prompt = self.REVIEW_PROMPT.format(
            query=ctx.query, draft=draft[:4000], schema=json_schema_hint(ReviewVerdict)   # 草稿截断到 4000 字
        )
        try:
            data = self.llm.invoke_structured(prompt, system_prompt=self.system_prompt)
            verdict = ReviewVerdict.model_validate(data)
            content = (
                f"评审{'通过' if verdict.passed else '不通过'}（{verdict.score:.2f}）\n"
                f"问题：{'; '.join(verdict.issues) or '无'}\n建议：{verdict.suggestion or '无'}"
            )
            return AgentOutput(
                agent=self.name, role=self.spec.role, content=content, success=True,
                duration_ms=int((time.time() - started) * 1000),
            )
        except Exception as e:
            _logger.warning(f"[{self.name}] 评审解析失败，按通过处理: {e}")
            return AgentOutput(
                agent=self.name, role=self.spec.role,
                content=f"评审未生效（解析失败: {e}），默认放行",
                success=True, duration_ms=int((time.time() - started) * 1000),
            )


# 【什么】构造默认团队的角色清单
# 【作用】提供「主管 + 检索 + 分析 + 审核」四件套，开箱即可跑一条完整链路
def build_default_team() -> list[AgentSpec]:
    """默认团队：1 主管 + 1 检索 + 1 分析 + 1 审核。

    【为什么默认只有 4 个人】
        Agent 数量不是越多越好：每多一个成员，主管的 prompt 就长一截，
        规划质量反而下降。4 人覆盖"找信息 → 算数据 → 审质量"三段，
        是性价比最高的起点；要扩成员在 TeamConfig 里加即可。
    """
    return [
        AgentSpec(
            name="supervisor", role=AgentRole.SUPERVISOR,
            description="拆解任务、派活、汇总最终答案", temperature=0.2,
        ),
        # 【为什么给这两个角色开委派资格】
        #   他们是最容易"缺一块"的人：analyst 常常没有原始数据、
        #   researcher 常常需要别人验算。开关（delegation_enabled）默认关闭，
        #   所以这里只是"具备能力"，不打开就完全不会发生委派、也不会多花一次调用。
        AgentSpec(
            name="researcher_1", role=AgentRole.RESEARCHER,
            description="查资料、找事实依据", tools=["data_query"], multimodal=True,
            can_delegate=True,
        ),
        AgentSpec(
            name="analyst_1", role=AgentRole.ANALYST,
            description="做计算、对比、归因", tools=["calculator"],
            can_delegate=True,
        ),
        AgentSpec(
            name="reviewer_1", role=AgentRole.REVIEWER,
            description="审核最终答案质量", temperature=0.1,
        ),
    ]


# 【什么】按角色实例化对应的 Agent 类
# 【作用】角色到类的映射集中在这一处，其余 Agent 角色默认落到 WorkerAgent
def build_agent(spec: AgentSpec, llm=None, tool_registry=None, gate=None) -> BaseAgent:
    """按角色选出对应的 Agent 类（角色 → 类的映射只在这一处）。"""
    if spec.role == AgentRole.SUPERVISOR:
        return SupervisorAgent(spec, llm, tool_registry, gate)
    if spec.role == AgentRole.REVIEWER:
        return ReviewerAgent(spec, llm, tool_registry, gate)
    return WorkerAgent(spec, llm, tool_registry, gate)
