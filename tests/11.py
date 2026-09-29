from modules.harness.models import ToolPolicy, RiskLevel
from modules.harness.harness import ToolHarness
from modules.tools.data_query import DataQueryTool

t = DataQueryTool()
print("原始声明:", t.name, "|", t.tool_type, "|", t.capability)

h1 = ToolHarness(policy=ToolPolicy(default_risk=RiskLevel.LOW))
cap1 = h1.register(t)
print("无覆盖:", cap1.to_dict())

h2 = ToolHarness(policy=ToolPolicy(default_risk=RiskLevel.LOW, overrides={"data_query": "high"}))
cap2 = h2.register(t)
print("有覆盖:", cap2.to_dict())
print("台账:", {k: v.risk.value for k, v in h2._capabilities.items()})
d = h2.policy.decide(cap2)
print("裁决:", d.action, "|", d.reason)
