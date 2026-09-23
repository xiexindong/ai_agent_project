"""
llama-index / crewai 桥接演示（两个库不装也能跑）
=================================================
    python examples/framework_bridge_demo.py

【这个脚本验证什么】
    验证的是"我们写的适配层对不对"：映射逻辑、权限表达式翻译、
    结果回填 RunResult、以及库缺失时的降级路径。
    不验证框架本身（那需要真装库 + 真模型 Key）。

【没装库怎么还能跑】
    往 sys.modules 里注入两个假模块（接口与真库一致），
    于是适配层里 `from crewai import Agent, Task, Crew` 会命中假模块，
    真实代码路径被完整走一遍 —— 包括构造参数、依赖顺序、kickoff、结果回填。
    装了真库时脚本自动跳过注入，直接走真库。

【真装库之后怎么用】
    pip install -r requirements-extras.txt
    .env: RAG_BACKEND=llamaindex  /  MULTIAGENT_BACKEND=crewai
"""
import os
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.common.exceptions import RAGException
from modules.multiagent.models import PlanResult, SubTask, TeamConfig
from modules.multiagent.agents import build_default_team
from modules.multiagent.crewai_backend import CrewAIBackend, is_available as crewai_ready
from modules.rag.llamaindex_adapter import (
    LlamaIndexRAG, parse_expr, is_available as li_ready,
)

# 【什么】假模块的行为记录本
# 【作用】FakeAgent/FakeTask/FakeCrew/FakeRetriever 把构造参数写进来，供人核对依赖顺序
RECORD = {
    "agents": [],              # 每次 new FakeAgent 时记一份关键字参数
    "tasks": [],               # 每次 new FakeTask 时记一份，含依赖 Task 的 id 列表
    "crew_kwargs": {},         # new FakeCrew 的参数（如 process）
    "retriever_kwargs": {},    # as_retriever 的参数（如 filters / similarity_top_k）
}


# ==========================================================================
# 第 1 段：可用性检测
# ==========================================================================
# 【什么】第 1 段：可选框架可用性检测
# 【作用】打印 llama-index 与 crewai 是否安装，未安装时说明各自降级到哪条自研路径
def demo_availability():
    print("=" * 68)
    print("第 1 段  可选框架可用性（未安装 → 自动降级，主流程不受影响）")
    print("=" * 68)
    print(f"  llama-index : {'已安装' if li_ready() else '未安装 → RAG 走自研 Milvus+BGE'}")
    print(f"  crewai      : {'已安装' if crewai_ready() else '未安装 → 编排走自研 Harness'}")


# ==========================================================================
# 第 2 段：权限表达式翻译（安全核心，纯 Python 可单测）
# ==========================================================================
# 【什么】第 2 段：Milvus 权限表达式翻译
# 【作用】4 条样例逐个 parse_expr，成功打印结构、失败打印拒绝原因
def demo_expr():
    print("\n" + "=" * 68)
    print("第 2 段  Milvus 权限表达式 → llama-index 过滤条件")
    print("=" * 68)
    cases = [
        'kb_code in ["kb_finance","kb_public"]',
        'level == "public" and kb_code in ["kb_public"]',
        "",
        'score > 0.5',
    ]
    for expr in cases:
        try:
            parsed = parse_expr(expr)
            print(f"  输入 {expr!r}")
            print(f"  输出 {parsed}")
        except RAGException as e:
            print(f"  输入 {expr!r}")
            print(f"  拒绝 {type(e).__name__}: {e}")
    print("\n  注意：翻译不了就抛异常拒绝查询 —— 过滤失效等于越权文档进上下文，")
    print("        宁可这次查不出东西，也不能让不该看的内容被召回。")


# ==========================================================================
# 第 3 段：团队配置 → crewai IR（不依赖库）
# ==========================================================================
# 【什么】第 3 段：团队配置转成 crewai 中间表示
# 【作用】打印 process、每个 Agent 与 Task 的关键字段，并返回 backend 与 plan 供后面复用
def demo_crew_spec():
    print("\n" + "=" * 68)
    print("第 3 段  自研 PlanResult → crewai 配置（中间表示 IR）")
    print("=" * 68)
    plan = PlanResult(
        thought="先查数据再算环比",
        subtasks=[
            SubTask(id="t1", assignee="researcher_1", instruction="查 Q3/Q4 营收"),
            SubTask(id="t2", assignee="analyst_1", instruction="算环比增长率", depends_on=["t1"]),
        ],
    )
    backend = CrewAIBackend(TeamConfig(agents=build_default_team()))
    spec = backend.build_crew_spec(plan, mm_text="【图片理解结果】柱状图显示 Q4 高于 Q3")
    print(f"  process : {spec['process']}")
    print(f"  warnings: {spec['warnings'] or '无'}")
    for a in spec["agents"]:
        print(f"  Agent {a['key']}: role={a['role']} | 委派={a['allow_delegation']} | 工具={a['tools']}")
    for t in spec["tasks"]:
        print(f"  Task  {t['key']}: agent={t['agent']} | context={t['context']}")
        print(f"         描述前 40 字: {t['description'][:40]}")
    print("\n  只给被派活的成员建 Agent（t1/t2 用了 2 人，不建闲置的 reviewer）:")
    print(f"    团队共 {len(build_default_team())} 人 → 实际建 {len(spec['agents'])} 个 Agent")
    return backend, plan


# ==========================================================================
# 第 4 段：注入假 crewai，跑完整 run()
# ==========================================================================
# 【什么】本文件用的极简假 LLM
# 【作用】只实现主管规划要用的两个接口，固定返回 2 个子任务的计划
class MockLLM:
    """极简假 LLM：只给主管的规划用。"""

    # 【什么】普通文本调用
    # 【作用】固定返回一句模拟结论，不联网不花钱
    def invoke(self, prompt: str, system_prompt: str = "", temperature: float = 0.3,
               max_tokens: int = 2048) -> str:
        return "（模拟回答）已完成分析，结论是环比上升 18%。"

    # 【什么】结构化输出调用
    # 【作用】返回 t1 查数、t2 算环比（依赖 t1）的固定计划
    def invoke_structured(self, prompt: str, system_prompt: str = "", temperature: float = 0.2) -> dict:
        return {
            "thought": "先查后算", "direct_answer": "", "needs_review": False,
            "subtasks": [
                {"id": "t1", "assignee": "researcher_1", "instruction": "查 Q3/Q4 营收",
                 "depends_on": [], "priority": 3},
                {"id": "t2", "assignee": "analyst_1", "instruction": "算环比增长率",
                 "depends_on": ["t1"], "priority": 2},
            ],
        }


def install_fake_crewai():
    """注入与真库接口一致的假 crewai，用于验证适配层的真实代码路径。"""

    class FakeAgent:
        def __init__(self, **kw):
            RECORD["agents"].append(kw)

    class FakeTask:
        def __init__(self, **kw):
            # context 必须是 Task 对象（不是字符串 id）—— 记下来供人检查依赖顺序
            RECORD["tasks"].append({**kw, "context_ids": [id(c) for c in kw.get("context") or []]})

    class FakeOutput:
        raw = "（模拟 crewai 汇总）Q4 营收 1475 万，环比 +18%。"
        tasks_output = [
            types.SimpleNamespace(description="查 Q3/Q4 营收", raw="Q3=1250万，Q4=1475万", name=""),
            types.SimpleNamespace(description="算环比增长率", raw="环比 +18%", name=""),
        ]

    class FakeCrew:
        def __init__(self, **kw):
            RECORD["crew_kwargs"] = kw

        def kickoff(self, inputs: dict = None):
            return FakeOutput()

    crewai = types.ModuleType("crewai")
    crewai.Agent, crewai.Task, crewai.Crew = FakeAgent, FakeTask, FakeCrew
    crewai.Process = types.SimpleNamespace(sequential="sequential", hierarchical="hierarchical")
    tools_mod = types.ModuleType("crewai.tools")

    class FakeCrewBaseTool:
        """假的 crewai BaseTool：只提供被继承的能力。"""

    tools_mod.BaseTool = FakeCrewBaseTool
    crewai.tools = tools_mod
    sys.modules["crewai"], sys.modules["crewai.tools"] = crewai, tools_mod
    return crewai


def demo_crew_run():
    print("\n" + "=" * 68)
    print("第 4 段  注入假 crewai → 跑完整 run()（验证构造参数与结果回填）")
    print("=" * 68)
    fake_used = not crewai_ready()
    if fake_used:
        install_fake_crewai()
        print("  （未安装真 crewai，已注入假模块走通真实代码路径）")

    backend = CrewAIBackend(TeamConfig(agents=build_default_team()), llm=MockLLM())
    result = backend.run("对比 Q3 与 Q4 营收", session_id="bridge-demo")

    print(f"  status      : {result.status.value}")
    print(f"  最终答案    : {result.final_answer[:52]}")
    print(f"  产出数      : {len(result.outputs)}（每个 Task 回填成一个 AgentOutput）")
    for o in result.outputs:
        print(f"    - {o.task_id} / {o.agent}: {o.content[:24]}")
    print(f"  trace       : {[e.kind for e in result.trace]}")
    if fake_used:
        print(f"  建 Agent 数 : {len(RECORD['agents'])} | 建 Task 数: {len(RECORD['tasks'])}")
        print(f"  Task 依赖   : {[ (t['context_ids'] != []) for t in RECORD['tasks'] ]}"
              "  → [False, True] 表示 t2 拿到了 t1 的 Task 对象")
        print(f"  Crew 参数   : process={RECORD['crew_kwargs'].get('process')}")
    sys.modules.pop("crewai", None)
    sys.modules.pop("crewai.tools", None)


# ==========================================================================
# 第 5 段：注入假 llama_index，跑检索
# ==========================================================================
def install_fake_llamaindex():
    """注入与 llama_index.core 接口一致的假模块。"""

    class FakeNode:
        def __init__(self, text, score, meta):
            self.node = types.SimpleNamespace(text=text, metadata=meta)
            self.score = score

    class FakeRetriever:
        def __init__(self, **kw):
            RECORD["retriever_kwargs"] = kw

        def retrieve(self, query):
            return [
                FakeNode("Q3 营收 1250 万", 0.91, {"file_name": "sales.md", "kb_code": "kb_finance"}),
                FakeNode("Q4 营收 1475 万", 0.88, {"file_name": "sales.md", "kb_code": "kb_finance"}),
            ]

    class FakeStorageContext:
        @classmethod
        def from_defaults(cls, **kw):
            return cls()

        def persist(self, **kw):
            pass

    class FakeIndex:
        def __init__(self):
            self.storage_context = FakeStorageContext()

        @classmethod
        def from_documents(cls, docs):
            return cls()

        def insert(self, doc):
            pass

        def as_retriever(self, **kw):
            return FakeRetriever(**kw)

    class FakeReader:
        def __init__(self, **kw):
            pass

        def load_data(self):
            return [types.SimpleNamespace(text="Q3=1250, Q4=1475")]

    core = types.ModuleType("llama_index.core")
    core.VectorStoreIndex, core.SimpleDirectoryReader = FakeIndex, FakeReader
    core.StorageContext, core.Settings = FakeStorageContext, types.SimpleNamespace()
    core.load_index_from_storage = lambda storage: FakeIndex()
    vs = types.ModuleType("llama_index.core.vector_stores")
    vs.FilterOperator = types.SimpleNamespace(IN="in", EQ="==", NE="!=")
    vs.MetadataFilter = lambda key, value, operator: {"key": key, "value": value, "op": operator}
    vs.MetadataFilters = lambda filters: {"filters": filters}
    pkg = types.ModuleType("llama_index")
    pkg.core = core
    sys.modules.update({
        "llama_index": pkg, "llama_index.core": core, "llama_index.core.vector_stores": vs,
    })


def demo_llamaindex():
    print("\n" + "=" * 68)
    print("第 5 段  注入假 llama_index → 跑入库与检索（含权限过滤）")
    print("=" * 68)
    fake_used = not li_ready()
    if fake_used:
        install_fake_llamaindex()
        print("  （未安装真 llama-index，已注入假模块走通真实代码路径）")

    persist = os.path.join(tempfile.gettempdir(), "li_demo_index")
    rag = LlamaIndexRAG(persist_dir=persist)

    tmp = os.path.join(tempfile.gettempdir(), "li_demo_doc.md")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("Q3 营收 1250 万\nQ4 营收 1475 万\n")
    n = rag.ingest_document(tmp)
    print(f"  入库文档数: {n}")

    hits = rag.retrieve("Q4 营收是多少", expr='kb_code in ["kb_finance"]')
    print(f"  检索命中  : {len(hits)} 条，字段 = {sorted(hits[0].keys()) if hits else '无'}")
    print(f"  过滤条件  : {RECORD['retriever_kwargs'].get('filters')}")
    print(f"  top_k     : {RECORD['retriever_kwargs'].get('similarity_top_k')}")
    print("\n  上下文拼接（字段与自研 RAGChain 完全一致，上层无需改代码）:")
    print("  " + rag.build_context("Q4 营收", expr='kb_code in ["kb_finance"]').replace("\n", "\n  ")[:160])

    os.remove(tmp)
    for m in ("llama_index", "llama_index.core", "llama_index.core.vector_stores"):
        sys.modules.pop(m, None)


def main():
    print("#" * 68)
    print("#  llama-index / crewai 桥接演示（未装库也能跑，装了自动走真库）")
    print("#" * 68)
    demo_availability()
    demo_expr()
    demo_crew_spec()
    demo_crew_run()
    demo_llamaindex()
    print("\n" + "=" * 68)
    print("装真库：pip install -r requirements-extras.txt")
    print("切后端：.env 里 RAG_BACKEND=llamaindex 或 MULTIAGENT_BACKEND=crewai")
    print("        默认 native —— 后端切换是迁移级操作，不靠默认值悄悄发生。")
    print("=" * 68)


if __name__ == "__main__":
    main()
