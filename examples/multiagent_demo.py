"""
多模态 + 多 Agent + Harness 演示脚本
====================================
零外部服务依赖（不联网、不需要 API Key、不需要数据库），直接跑：
    python examples/multiagent_demo.py

【演示顺序】
    1. 多模态输入：图片/文件片段的 pydantic 校验与 OpenAI Vision 报文转换
    2. 多 Agent 编排：supervisor 拆任务 → 并行 worker（含工具调用）→ 汇总 → 终审
    3. 直接答路径：简单问题不开团队，省掉整条链路的 token
    4. Harness 评测：跑 3 条用例出通过率（可进 CI 的那种评测）
    5. 依赖分批：topo_batches 的输入输出对照

【为什么用 MockLLM】
    这个脚本验证的是"编排链路对不对"（谁派给谁、什么时候并行、失败怎么降级），
    不是"模型回答得好不好"。用假 LLM 把模型输出固定住，
    每次跑出来的结构完全一致 —— 排查编排 bug 时才分得清是模型漂了还是代码错了。
    要接真模型：把 MockLLM() 换成 LLMClient()（见文件末尾"接真模型"）。

【接真模型需要什么】
    .env 里配好 DEEPSEEK_API_KEY（文本模型）+ VISION_API_KEY / VISION_ENABLED=true（视觉）
"""
import os
import sys
import tempfile
from pathlib import Path

# 让脚本在任意目录下都能 import 项目模块（否则从 examples/ 直接跑会报 No module named 'modules'）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError

from modules.multimodal.media import FilePart, ImagePart, MultiModalInput, TextPart
from modules.multiagent.models import EvalCase, SubTask
from modules.multiagent.harness import AgentHarness, build_harness
from modules.tools.base_tool import BaseTool, ToolResult
from modules.tools.tool_registry import ToolRegistry

# 1x1 像素的合法 PNG（base64），用于演示图片片段，避免依赖 pillow 造图
PNG_1PX_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8AAAwAB/wFbArTNAAAAAElFTkSuQmCC"
)


# ==========================================================================
# 假 LLM：按调用类型返回固定结构
# ==========================================================================
class MockLLM:
    """演示用 LLM：invoke / invoke_structured / invoke_with_tools 三个接口都实现。"""

    PLAN_JSON = {
        "thought": "需要先查事实再算环比，最后审核，t2 依赖 t1 的数据",
        "subtasks": [
            {"id": "t1", "assignee": "researcher_1",
             "instruction": "查 2024 年 Q4 与 Q3 的营收数据", "depends_on": [], "priority": 3},
            {"id": "t2", "assignee": "analyst_1",
             "instruction": "计算 Q4 相对 Q3 的环比增长率", "depends_on": ["t1"], "priority": 2},
        ],
        "direct_answer": "",
        "needs_review": True,
    }
    REVIEW_JSON = {"passed": True, "score": 0.86, "issues": [], "suggestion": "可补充数据来源"}

    def __init__(self):
        self.tool_round = 0

    def invoke(self, prompt: str, system_prompt: str = "", temperature: float = 0.3,
               max_tokens: int = 2048) -> str:
        head = " ".join(prompt.split())[:36]
        return f"（模拟回答）针对「{head}」：已完成分析，结论是环比上升 12%，建议持续观察。"

    def invoke_structured(self, prompt: str, system_prompt: str = "", temperature: float = 0.2) -> dict:
        # 按 prompt 里出现的字段名区分"规划"和"评审"两种结构化调用
        return self.PLAN_JSON if "subtasks" in prompt else self.REVIEW_JSON

    def invoke_with_tools(self, messages: list, tools: list, temperature: float = 0.3) -> dict:
        """第一轮返回工具调用，第二轮返回结论 —— 演示 Function Calling 循环确实转起来了。"""
        self.tool_round += 1
        if tools and self.tool_round == 1:
            return {
                "content": "",
                "tool_calls": [{"name": "calculator", "arguments": {"expression": "1250*1.18"},
                                "id": "call_1"}],
            }
        return {"content": "（模拟回答）已完成计算，Q4 营收 1475 万，环比 +18%。", "tool_calls": None}


# ==========================================================================
# 一个纯本地工具：让 worker 的工具调用路径有东西可调
# ==========================================================================
class DemoCalculator(BaseTool):
    """演示用计算器（只支持加减乘除，纯本地不联网）。"""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="计算数学表达式，如 1250*1.18",
            parameters={"properties": {"expression": {"type": "string"}}, "required": ["expression"]},
        )

    def execute(self, **kwargs) -> ToolResult:
        expr = kwargs.get("expression", "")
        try:
            value = eval(expr, {"__builtins__": {}}, {})  # demo 专用：仅本地演示，生产请用 AST 白名单
            return ToolResult(success=True, result=value)
        except Exception as e:
            return ToolResult(success=False, error=f"表达式非法: {e}")


# ==========================================================================
# 第 1 段：多模态输入
# ==========================================================================
def demo_multimodal():
    print("=" * 68)
    print("第 1 段  多模态输入：pydantic 校验 + 转成 Vision 报文")
    print("=" * 68)

    mm = MultiModalInput(
        instruction="这张图里的销售额趋势如何？",
        parts=[
            TextPart(text="重点看 Q4 的柱子"),
            ImagePart(source=f"data:image/png;base64,{PNG_1PX_B64}", caption="季度销售柱状图"),
        ],
    )
    print(f"图片张数: {mm.image_count}")
    print(f"纯文本降级视图: {mm.text_only}")
    print("发给视觉模型的 content 数组:")
    for block in mm.to_openai_content():
        url = block.get("image_url", {}).get("url", "")
        print(f"  - {block['type']}: {url[:48] + '...' if url else block.get('text', '')[:48]}")

    # 文件片段：临时写一个 csv 再读，读完删掉
    tmp = os.path.join(tempfile.gettempdir(), "demo_sales.csv")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("quarter,revenue\nQ3,1250\nQ4,1475\n")
    file_part = FilePart(path=tmp)
    print(f"\n文件片段: name={file_part.name} bytes={file_part.bytes_size}")
    print(f"抽取内容: {file_part.content.strip()}")
    os.remove(tmp)

    # 校验失败的两个例子（错误必须在入口暴露，而不是花掉一次远端调用后才发现）
    for bad, why in [
        ({"source": "data:image/tiff;base64,AAAA"}, "不在白名单的图片类型"),
        ({"source": "/no/such/file.png"}, "本地文件不存在"),
    ]:
        try:
            ImagePart(**bad)
            print(f"  [未拦截] {why}")
        except ValidationError as e:
            print(f"  [已拦截] {why}: {e.errors()[0]['msg']}")


# ==========================================================================
# 第 2 段：多 Agent 编排
# ==========================================================================
def demo_orchestration(harness: AgentHarness):
    print("\n" + "=" * 68)
    print("第 2 段  多 Agent 编排：拆任务 → 并行执行 → 汇总 → 终审")
    print("=" * 68)

    result = harness.run("对比公司 Q3 与 Q4 的营收，给出环比结论", session_id="demo-001")
    plan = result.plan
    print(f"主管思路: {plan.thought}")
    print(f"子任务 {len(plan.subtasks)} 个:")
    for t in plan.subtasks:
        dep = f"（依赖 {t.depends_on}）" if t.depends_on else "（可立即开始）"
        print(f"  - {t.id} → {t.assignee}: {t.instruction} {dep}")

    print("\n执行产出:")
    for out in result.outputs:
        flag = "成功" if out.success else "失败"
        extra = f" | 工具 {len(out.tool_calls)} 次" if out.tool_calls else ""
        print(f"  [{flag}] {out.agent}({out.role.value}) 步数={out.steps} 耗时={out.duration_ms}ms{extra}")
        print(f"         {out.content[:60]}")

    print(f"\n最终答案: {result.final_answer[:90]}")
    if result.review:
        print(f"终审结论: {result.review.content[:60]}")
    print(f"状态: {result.status.value} | 总耗时: {result.duration_ms}ms")

    print("\nTrace 时间线（排查全靠它）:")
    base = result.trace[0].ts if result.trace else 0
    for ev in result.trace:
        print(f"  +{ev.ts - base:>5}ms  {ev.kind:<12} {ev.agent or '-':<14} {ev.message[:44]}")


# ==========================================================================
# 第 3 段：直接答路径 + 失败降级
# ==========================================================================
def demo_direct_and_failure(harness: AgentHarness):
    print("\n" + "=" * 68)
    print("第 3 段  直接答路径与失败降级")
    print("=" * 68)

    # 简单问题：主管不拆任务，省掉整条团队链路
    direct_plan = {
        "thought": "寒暄，无需拆任务", "subtasks": [], "direct_answer": "你好，有什么可以帮你？",
        "needs_review": False,
    }
    original = harness.supervisor.llm.invoke_structured
    harness.supervisor.llm.invoke_structured = lambda *a, **k: direct_plan
    result = harness.run("你好")
    harness.supervisor.llm.invoke_structured = original
    print(f"是否直接答: {result.plan.is_direct} | 子任务数: {len(result.plan.subtasks)}")
    print(f"答案: {result.final_answer}")
    print(f"产出数: {len(result.outputs)}（不派活 = 不花 worker 的 token）")

    # 派给不存在的 Agent：不能让整次运行崩，只能降级成该任务失败
    bad_plan = {
        "thought": "演示错误分派", "direct_answer": "", "needs_review": False,
        "subtasks": [{"id": "t1", "assignee": "ghost_agent", "instruction": "查数据",
                      "depends_on": [], "priority": 1}],
    }
    harness.supervisor.llm.invoke_structured = lambda *a, **k: bad_plan
    result = harness.run("派给一个不存在的成员会怎样")
    harness.supervisor.llm.invoke_structured = original
    print(f"\n错误分派结果: status={result.status.value} error={result.error}")
    print(f"单个失败记录的 error: {result.outputs[0].error if result.outputs else '（无）'}")


# ==========================================================================
# 第 4 段：评测
# ==========================================================================
def demo_evaluate(harness: AgentHarness):
    print("\n" + "=" * 68)
    print("第 4 段  Harness 评测：关键词 + 完成率 + 速度（可进 CI）")
    print("=" * 68)
    cases = [
        EvalCase(name="营收对比", query="对比 Q3 与 Q4 营收", expected_keywords=["已完成", "结论"]),
        EvalCase(name="库存盘点", query="盘点当前库存", expected_keywords=["已完成"]),
        EvalCase(name="缺词用例", query="随便问一句", expected_keywords=["该词必然不出现"]),
    ]
    report = harness.evaluate(cases)
    print(f"通过: {report.pass_rate} | 平均分: {report.score}")
    for r in report.results:
        print(f"  - {r.case}: {'通过' if r.passed else '不通过'} 分数={r.score} 原因={r.reason}")


# ==========================================================================
# 第 5 段：依赖分批
# ==========================================================================
def demo_topo():
    print("\n" + "=" * 68)
    print("第 5 段  依赖分批：DAG 拓扑排序（同批并行）")
    print("=" * 68)
    tasks = [
        SubTask(id="t1", assignee="researcher_1", instruction="查数据"),
        SubTask(id="t2", assignee="analyst_1", instruction="算环比", depends_on=["t1"]),
        SubTask(id="t3", assignee="analyst_1", instruction="算同比", depends_on=["t1"]),
        SubTask(id="t4", assignee="reviewer_1", instruction="终审", depends_on=["t2", "t3"], priority=5),
    ]
    batches, cyclic = AgentHarness.topo_batches(tasks)
    print(f"输入 {len(tasks)} 个任务 → 输出 {len(batches)} 批，环/无效依赖: {cyclic or '无'}")
    for i, batch in enumerate(batches, 1):
        print(f"  第 {i} 批（并行 {len(batch)} 个）: {[t.id for t in batch]}")

    looped = [
        SubTask(id="a", assignee="analyst_1", instruction="A", depends_on=["b"]),
        SubTask(id="b", assignee="analyst_1", instruction="B", depends_on=["a"]),
    ]
    batches2, cyclic2 = AgentHarness.topo_batches(looped)
    print(f"\n循环依赖: 批数={len(batches2)} 被剔除的任务={cyclic2}")
    print("  → 不剔除就会 while 死循环（不是报错，是卡住，最难排查的那种故障）")


def main():
    print("#" * 68)
    print("#  多模态 + 多 Agent + Harness 演示（MockLLM，无需联网）")
    print("#" * 68)

    registry = ToolRegistry()
    registry.register(DemoCalculator())
    harness = build_harness(llm=MockLLM(), tool_registry=registry, max_workers=2)
    print(f"团队: {[a.name for a in harness.agents.values()]}")

    demo_multimodal()
    demo_orchestration(harness)
    demo_direct_and_failure(harness)
    demo_evaluate(harness)
    demo_topo()
    print("\n接真模型：把 MockLLM() 换成 LLMClient()，.env 配 DEEPSEEK_API_KEY；")
    print("         要开图片理解，再配 VISION_ENABLED=true 与 VISION_API_KEY。")


if __name__ == "__main__":
    main()
