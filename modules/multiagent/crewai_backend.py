"""
crewai 编排后端（可选，默认不启用）
==================================
【定位】
    自研 Harness 是默认编排；crewai 是"第二种编排实现"。
    两者对外返回**同一个 RunResult**，所以 API 的 response_model、
    前端解析、评测逻辑全部不用改 —— 这是适配层的全部价值。

【为什么用 crewai 而不是只留自研 Harness】
    自研的好处是可控（DAG 并行、超时、步数预算、trace 都是自己写的）；
    crewai 的好处是角色扮演的提示词工程成熟（role/goal/backstory 三件套
    对模型行为约束很有效）、社区工具生态现成。两个都留着，按场景切。

【能复用的一律复用，不重复造轮子】
    规划仍然用自研 SupervisorAgent.plan()：它已经处理了 schema 校验、
    失败重试、降级为单任务。crewai 只接管"执行 + 汇总"这一段。

【依赖警告（集成时最容易翻车的三点）】
    1. crewai 走 litellm 调模型，不吃我们自己的 LLMClient，
       需要单独配模型环境变量（如 OPENAI_API_KEY / DEEPSEEK_* 被 litellm 识别的形式）。
    2. crewai 的 Task 依赖靠 context=[上游Task对象] 传递，不是字符串 id，
       所以必须按拓扑序先建上游 Task 对象。
    3. crewai 不原生支持图片输入：多模态必须先转成文本再喂进去（本文件 _mm_text）。
"""
import time
from typing import Optional

from modules.common.logger import get_logger
from modules.multiagent.models import (
    AgentOutput,
    AgentRole,
    RunResult,
    RunStatus,
    TeamConfig,
    TraceEvent,
    TraceKind,
)
from modules.multiagent.agents import RunContext, SupervisorAgent, build_default_team

_logger = get_logger("crewai_backend")


# 【什么】探测 crewai 是否已安装
# 【作用】供后端工厂与 /agent/backends 探活；未安装返回 False，不抛导入错误
def is_available() -> bool:
    """检测 crewai 是否可用（供工厂与 /agent/backends 探活用）。"""
    try:
        import crewai  # noqa: F401  # type: ignore   # 可选依赖，未装时由 except 降级
        return True
    except Exception:
        return False


# 【什么】crewai 编排后端
# 【作用】把自研团队与计划映射成 crewai 的 Agent/Task/Crew，返回与自研一致的 RunResult
class CrewAIBackend:
    """把自研团队配置映射成 crewai 的 Agent/Task/Crew 并执行。"""

    # 【什么】装配 crewai 后端
    # 【作用】拆出主管与成员两类角色，规划仍复用自研 SupervisorAgent
    def __init__(self, config: Optional[TeamConfig] = None, llm=None, tool_registry=None):
        self.config = config or TeamConfig(agents=build_default_team())
        self.llm = llm
        self.tools = tool_registry
        specs = [s for s in self.config.agents if s.enabled]   # 只装配 enabled=True 的角色
        self.supervisor_spec = next((s for s in specs if s.role == AgentRole.SUPERVISOR), None)
        self.supervisor = (
            SupervisorAgent(self.supervisor_spec, llm=llm, tool_registry=tool_registry)
            if self.supervisor_spec else None
        )
        self.members = [s for s in specs if s.role != AgentRole.SUPERVISOR]   # 可派活的执行者清单

    # ------------------------------------------------------------------
    # 【中间表示 IR】先把团队和任务映射成纯 dict，再据此构造 crewai 对象
    #   为什么要多这一层：
    #   1. 映射逻辑可在没装 crewai 的情况下验证与打印（进 CI 不用装重依赖）
    #   2. 构造真实对象需要按顺序解析依赖（Task.context 要传对象不是 id），
    #      IR 保留了 id 引用关系，materialize 阶段再换成对象引用
    #
    # 【输入输出对照】
    #   输入：plan.subtasks = [SubTask(id="t1", assignee="researcher_1",
    #                                 instruction="查Q4营收", depends_on=[])]
    #   输出：{"tasks":[{"key":"t1","agent":"researcher_1","context":[],
    #                   "description":"查Q4营收","expected_output":"..."}]}
    # ------------------------------------------------------------------
    # 【什么】把计划映射成 crewai 的中间表示（纯 dict）
    # 【作用】不依赖 crewai 是否安装，便于单测与调试；assignee 未注册时记进 warnings
    def build_crew_spec(self, plan, mm_text: str = "") -> dict:
        """把 PlanResult 映射成 crewai 配置（IR，不依赖 crewai 是否安装）。"""
        used = {t.assignee for t in plan.subtasks}   # 本次实际被派活的 Agent 名集合
        agents = []
        for m in self.members:
            if m.name not in used:
                continue
            agents.append({
                "key": m.name,
                "role": f"{m.role.value}（{m.name}）",
                "goal": m.description or f"完成分配给 {m.name} 的子任务",
                "backstory": m.system_prompt or m.description or "",
                "tools": list(m.tools),
                # 只有主管能委派：worker 之间互相委派会出现"甩锅死循环"，
                # 且每次委派都是一轮新的 LLM 调用，成本不可控
                "allow_delegation": False,
            })

        tasks = []
        warnings = []
        for t in plan.subtasks:
            if t.assignee not in used:
                warnings.append(f"任务 {t.id} 的 assignee 未注册: {t.assignee}")
            desc = t.instruction
            if mm_text:
                desc = f"{mm_text}\n\n{t.instruction}"
            tasks.append({
                "key": t.id,
                "agent": t.assignee,
                "description": desc,
                "expected_output": "一段可直接交给主管汇总的结论文本（含关键数据）",
                "context": list(t.depends_on),
            })

        return {
            # 依赖已经在我们的 DAG 里排过序，这里用 sequential 按拓扑序喂进去：
            # hierarchical 会让 crewai 自己再分派一次，与我们的调度打架
            "process": "sequential",
            "agents": agents,
            "tasks": tasks,
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # 【多模态转文本】crewai 不认图片，图片必须先变成文字
    #   走的是同一个 VisionClient，视觉不可用时它内部已降级成占位文本，
    #   不会出现" crewai 崩了"的情况 —— 降级在最底层统一做一次就够。
    # ------------------------------------------------------------------
    # 【什么】把多模态输入转成文本描述
    # 【作用】crewai 不认图片，先把图译成文字；视觉不可用时退回占位文本
    def _mm_text(self, mm) -> str:
        if not mm or not mm.image_count:
            return ""
        try:
            from modules.multimodal.vision import get_vision_client
            desc = get_vision_client().invoke(mm, system_prompt="用中文简述图片中与问题相关的信息")
            return f"【图片理解结果】{desc}"
        except Exception as e:
            _logger.warning(f"图片理解降级（改用占位文本）: {e}")
            return f"【图片（未能识别）】{mm.text_only}"

    # ------------------------------------------------------------------
    # 【工具适配】自研 BaseTool → crewai 工具
    #   crewai 的工具要一个 pydantic args_schema，而自研工具只有一份
    #   JSON Schema（parameters）。这里按 JSON type 现场生成一个模型。
    #   生产环境建议给高频工具手写 schema，自动映射只保证"能调通"。
    # ------------------------------------------------------------------
    # 【什么】把自研 BaseTool 包装成 crewai 工具
    # 【作用】按 JSON Schema 现场生成 pydantic args 模型，让 crewai 能校验并调用
    def _to_crew_tool(self, base_tool):
        from crewai.tools import BaseTool as CrewBaseTool   # type: ignore   # 可选依赖
        from pydantic import create_model

        type_map = {"string": str, "integer": int, "number": float, "boolean": bool}   # JSON type → Python 类型
        props = (base_tool.parameters or {}).get("properties", {}) or {}
        required = set((base_tool.parameters or {}).get("required", []) or [])
        fields = {}
        for name, schema in props.items():
            py_type = type_map.get((schema or {}).get("type", "string"), str)
            fields[name] = (py_type, ... if name in required else None)
        args_model = create_model(f"{base_tool.name}Args", **fields) if fields else None

        # 【什么】crewai 工具的实际执行体
        # 【作用】转调自研工具；失败时返回错误文本而不是抛异常，避免打断 crew 执行
        def _run(self, **kwargs):
            result = base_tool.run(kwargs)
            return result.result if result.success else f"工具调用失败: {result.error}"

        return type(
            f"CrewTool_{base_tool.name}",
            (CrewBaseTool,),
            {"name": base_tool.name, "description": base_tool.description,
             "args_schema": args_model, "_run": _run},
        )()

    # 【什么】把 IR 变成真实的 crewai Agent/Task/Crew
    # 【作用】按拓扑顺序建 Task，保证 context 里引用的都是已建好的上游对象
    def _materialize(self, spec: dict):
        """IR → 真实 crewai 对象。按拓扑序建 Task，保证 context 里都是已建好的对象。"""
        from crewai import Agent, Task, Crew, Process   # type: ignore   # 可选依赖

        agent_objs = {}
        for a in spec["agents"]:
            tools = []
            for tname in a["tools"]:
                bt = self.tools.get_tool(tname) if self.tools else None
                if bt is None:
                    continue
                try:
                    tools.append(self._to_crew_tool(bt))
                except Exception as e:
                    _logger.warning(f"工具 {tname} 转 crewai 工具失败（跳过）: {e}")
            agent_objs[a["key"]] = Agent(
                role=a["role"], goal=a["goal"], backstory=a["backstory"],
                tools=tools, allow_delegation=a["allow_delegation"], verbose=False,
            )

        task_objs = {}
        for t in spec["tasks"]:
            task_objs[t["key"]] = Task(
                description=t["description"],
                expected_output=t["expected_output"],
                agent=agent_objs.get(t["agent"]),
                context=[task_objs[d] for d in t["context"] if d in task_objs],
            )
        return Crew(
            agents=list(agent_objs.values()),
            tasks=list(task_objs.values()),
            process=Process.sequential,
            verbose=False,
        )

    # 【什么】执行一次团队运行（crewai 版）
    # 【作用】复用自研规划、crewai 执行；任何异常都转成 FAILED 结果，不向外抛
    def run(self, query: str, mm=None, session_id: str = "", run_id: str = "") -> RunResult:
        """
        执行一次团队运行（crewai 版）。返回值与自研 harness 完全一致：RunResult。
        """
        started = time.time()
        result = RunResult(query=query, multimodal=bool(mm and mm.image_count))
        if run_id:
            result.run_id = run_id

        # 【什么】本轮运行的 trace 写入函数
        # 【作用】把事件追加到 result.trace，结构与自研 harness 的 _emit 保持一致
        def emit(kind: TraceKind, message: str, agent: str = "", **payload):
            result.trace.append(TraceEvent(kind=kind, message=message, agent=agent, payload=payload))

        try:
            if not is_available():
                raise RuntimeError("crewai 未安装，安装：pip install -r requirements-extras.txt")
            if self.supervisor is None:
                raise RuntimeError("团队缺少 supervisor，无法规划")

            # 第 1 步：复用自研主管做规划（schema 校验/重试/降级都在里面）
            ctx = RunContext(query=query, mm=mm, session_id=session_id)
            plan = self.supervisor.plan(ctx, self.members)
            result.plan = plan
            emit("plan", plan.thought or f"拆出 {len(plan.subtasks)} 个子任务", self.supervisor.name)

            if plan.is_direct:
                result.final_answer = plan.direct_answer
                result.status = RunStatus.SUCCESS
                result.duration_ms = int((time.time() - started) * 1000)
                emit("done", "主管判定无需拆任务，直接作答")
                return result

            # 第 2 步：IR → crewai 对象 → kickoff
            mm_text = self._mm_text(mm)
            spec = self.build_crew_spec(plan, mm_text=mm_text)
            for w in spec["warnings"]:
                _logger.warning(w)
            emit("dispatch", f"crewai 编排: {len(spec['agents'])} 个 Agent / {len(spec['tasks'])} 个 Task")

            crew = self._materialize(spec)
            raw = crew.kickoff()
            answer = getattr(raw, "raw", None) or str(raw)

            # 第 3 步：把 crewai 的每个任务产出回填成 AgentOutput，
            # 这样 trace / 评测 / 前端展示跟自研链路长得一模一样
            result.outputs = self._fill_outputs(raw, spec["tasks"])

            result.final_answer = answer
            result.status = RunStatus.SUCCESS
            result.duration_ms = int((time.time() - started) * 1000)
            emit("done", f"status=success 耗时={result.duration_ms}ms")
            return result

        except Exception as e:
            _logger.error(f"crewai 后端运行失败: {type(e).__name__}: {e}")
            result.status = RunStatus.FAILED
            result.error = f"{type(e).__name__}: {e}"
            result.final_answer = "抱歉，团队执行失败，已记录错误。"
            result.duration_ms = int((time.time() - started) * 1000)
            emit("error", result.error)
            return result

    # ------------------------------------------------------------------
    # 【产出回填】crewai 的 CrewOutput 不告诉我们哪个产出对应哪个 Task id
    #   它的 tasks_output 里只有 description（我们传进去的任务描述），
    #   没有 id 字段。两种对应办法：
    #     1. 数量一致 → 按 Task 顺序对应（sequential 模式下产出顺序 = 任务顺序）
    #     2. 数量不一致（有任务被跳过/重试）→ 退回描述前 40 字匹配
    #   都对不上就留空并记日志：宁可内容为空，也不能张冠李戴把 A 的结论
    #   标成 B 的产出 —— 汇总时会据此写进最终答案，错配比空更难发现。
    # ------------------------------------------------------------------
    # 【什么】把 crewai 的产出回填成 AgentOutput 列表
    # 【作用】让 trace / 评测 / 前端看到的结构与自研链路完全一致
    @staticmethod
    def _fill_outputs(raw, spec_tasks: list) -> list:
        items = list(getattr(raw, "tasks_output", None) or [])
        outputs = []

        # 【什么】取单个产出的正文文本
        # 【作用】优先取 raw 字段，取不到再退回 str() 兜底
        def text_of(item) -> str:
            return getattr(item, "raw", "") or str(item)

        if items and len(items) == len(spec_tasks):
            for t, item in zip(spec_tasks, items):
                outputs.append(AgentOutput(agent=t["agent"], task_id=t["key"], content=text_of(item)))
            return outputs

        by_desc = {(getattr(i, "description", "") or "")[:40]: i for i in items}   # 按描述前 40 字做兜底匹配
        for t in spec_tasks:
            item = by_desc.get(t["description"][:40])
            if item is None:
                _logger.warning(f"crewai 产出无法对应任务 {t['key']}（留空，不猜测）")
            outputs.append(AgentOutput(
                agent=t["agent"], task_id=t["key"], content=text_of(item) if item else "",
            ))
        return outputs
