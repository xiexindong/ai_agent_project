"""
Agent 核心模块
===============
LangGraph 状态化 Agent：ReAct推理循环、Function Calling工具调用、状态管理、任务调度。
覆盖边界2（输出格式校验）、边界3（死循环拦截）、边界4（上下文压缩）。
"""
# 【什么】ReAct 主循环依赖的三件套
# 【作用】LLM 客户端、状态定义、ReAct 引擎，import 本包即可直接用
from modules.agent.llm import LLMClient
from modules.agent.state import AgentState, AgentStateDict
from modules.agent.react_agent import ReActAgent
# 【什么】图编排层与 L0 路由层
# 【作用】AgentGraph 管节点跳转，AgentRouter 管本轮工具名单
from modules.agent.graph import AgentGraph
from modules.agent.router import AgentRouter, RoutingDecision

# 【什么】对外导出的符号清单（共 7 个）
# 【作用】约定包的公开接口，from modules.agent import * 只暴露这些名字
__all__ = [
    "LLMClient", "AgentState", "AgentStateDict", "ReActAgent", "AgentGraph",
    "AgentRouter", "RoutingDecision",
]
