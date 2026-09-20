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

_logger = get_logger("multiagent")


class LLMProtocol(Protocol):
    """LLM 能力契约。真实实现是 LLMClient，测试/演示可换成 MockLLM。"""

    def invoke(self, prompt: str, system_prompt: str = "", temperature: float = 0.3,
               max_tokens: int = 2048) -> str: ...

    def invoke_structured(self, prompt: str, system_prompt: str = "",
                          temperature: float = 0.2) -> dict: ...

    def invoke_with_tools(self, messages: list, tools: list, temperature: float = 0.3) -> dict: ...


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


@dataclass
class RunContext:
    """一次运行中所有 Agent 共享的上下文（只读）。"""

    query: str = ""
    mm: Optional[MultiModalInput] = None
    upstream: dict[str, AgentOutput] = field(default_factory=dict)
    session_id: str = ""
    principal: Any = None


class ReviewVerdict(BaseModel):
    """Reviewer 的评审结论（pydantic 校验，避免模型返回自由文本）。"""

    passed: bool = Field(description="草稿是否通过")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="0-1 质量分")
    issues: list[str] = Field(default_factory=list, description="发现的问题列表")
    suggestion: str = Field(default="", description="改进建议")


class BaseAgent:
    """Agent 基类：持有 spec、LLM、工具注册器，负责一次执行的通用流程。"""

    def __init__(self, spec: AgentSpec, llm: Optional[LLMProtocol] = None, tool_registry=None):
        self.spec = spec
        self._llm = llm
        self.tools = tool_registry
        self.system_prompt = spec.system_prompt or DEFAULT_ROLE_PROMPTS.get(spec.role, "")

    @property
    def name(self) -> str:
        return self.spec.name

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

    def _tool_schemas(self) -> list:
        """按 spec.tools 白名单取工具的 Function Calling schema。"""
        if not self.tools or not self.spec.tools:
            return []
        try:
            return self.tools.get_openai_tools_schema(user_level="advanced", allowed=self.spec.tools)
        except Exception as e:
            _logger.warning(f"[{self.name}] 工具 schema 生成失败（按无工具执行）: {e}")
            return []

    def _run_with_tools(self, prompt: str, steps_budget: int, session_id: str = "") -> tuple[str, int, list]:
        """
        Function Calling 循环：最多跑 steps_budget 轮，拿到不带工具调用的回答就收。

        【返回】(最终文本, 实际步数, 工具调用日志)
        【为什么限制轮数】不限制的话，模型可能在"调工具→看结果→再调工具"里打转，
            这和单 Agent ReAct 的 max_iterations=10 是同一个问题的不同尺度。
        """
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

    def run(self, task: Optional[SubTask], ctx: RunContext) -> AgentOutput:
        """执行一次子任务（Supervisor 之外的所有 Agent 走这里）。"""
        started = time.time()
        instruction = task.instruction if task else ctx.query
        upstream_text = self._upstream_text(ctx)
        vision_text = self.describe_images(ctx)
        prompt = self._build_prompt(instruction, upstream_text, vision_text)

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
            )
        except Exception as e:
            _logger.error(f"[{self.name}] 执行失败: {type(e).__name__}: {e}")
            return AgentOutput.failed(
                self.name, task.id if task else "", f"{type(e).__name__}: {e}",
                role=self.spec.role,
                duration_ms=int((time.time() - started) * 1000),
            )

    def _upstream_text(self, ctx: RunContext) -> str:
        """把上游产出拼成可读文本（每条截断 1200 字，防上下文爆炸）。"""
        if not ctx.upstream:
            return ""
        blocks = [
            f"【上游产出 · {k}】\n{(o.content or o.error)[:1200]}"
            for k, o in ctx.upstream.items()
        ]
        return "\n\n".join(blocks)

    def _build_prompt(self, instruction: str, upstream_text: str, vision_text: str) -> str:
        parts = [f"## 你的任务\n{instruction}"]
        if vision_text:
            parts.append(f"## 图片解析结果\n{vision_text}")
        if upstream_text:
            parts.append(f"## 上游产出\n{upstream_text}")
        parts.append("## 要求\n直接给出你的产出内容，不要复述任务，不要写客套话。")
        return "\n\n".join(parts)


class SupervisorAgent(BaseAgent):
    """主管：负责拆任务（plan）与汇总结论（synthesize）。"""

    PLAN_PROMPT = (
        "## 用户问题\n{query}\n\n"
        "## 输入形态\n{modality}\n\n"
        "## 可用成员\n{roster}\n\n"
        "## 成员产出规则\n- 能并行的任务不要串行（不要写无谓的 depends_on）\n"
        "- 一个任务只派一个人\n- 简单问题直接给 direct_answer\n\n"
        "## 输出结构（严格遵守）\n{schema}\n\n"
        "只输出一个 JSON 对象，不要多余文字。"
    )

    SYNTH_PROMPT = (
        "## 用户问题\n{query}\n\n"
        "## 团队成员产出\n{outputs}\n\n"
        "## 要求\n"
        "- 综合所有产出，给出一份完整、可直接交给用户的答案\n"
        "- 有冲突时以有数据支撑的为准，并说明分歧\n"
        "- 成员失败的部分如实说明'该部分未能完成'\n"
        "- 末尾附【摘要】一行总结"
    )

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


class WorkerAgent(BaseAgent):
    """执行者：接一个子任务，可用工具，产出一段结论。"""


class ReviewerAgent(BaseAgent):
    """审核者：对汇总草稿打分，输出结构化结论。"""

    REVIEW_PROMPT = (
        "## 用户问题\n{query}\n\n"
        "## 待审草稿\n{draft}\n\n"
        "## 输出结构（严格遵守）\n{schema}\n\n"
        "只输出一个 JSON 对象。"
    )

    def review(self, ctx: RunContext, draft: str) -> AgentOutput:
        """评审草稿。解析失败不阻断主流程：返回 passed=True 的中性结论。"""
        started = time.time()
        prompt = self.REVIEW_PROMPT.format(
            query=ctx.query, draft=draft[:4000], schema=json_schema_hint(ReviewVerdict)
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
        AgentSpec(
            name="researcher_1", role=AgentRole.RESEARCHER,
            description="查资料、找事实依据", tools=["data_query"], multimodal=True,
        ),
        AgentSpec(
            name="analyst_1", role=AgentRole.ANALYST,
            description="做计算、对比、归因", tools=["calculator"],
        ),
        AgentSpec(
            name="reviewer_1", role=AgentRole.REVIEWER,
            description="审核最终答案质量", temperature=0.1,
        ),
    ]


def build_agent(spec: AgentSpec, llm=None, tool_registry=None) -> BaseAgent:
    """按角色选出对应的 Agent 类（角色 → 类的映射只在这一处）。"""
    if spec.role == AgentRole.SUPERVISOR:
        return SupervisorAgent(spec, llm, tool_registry)
    if spec.role == AgentRole.REVIEWER:
        return ReviewerAgent(spec, llm, tool_registry)
    return WorkerAgent(spec, llm, tool_registry)
