"""
流水线 HTTP 端点
================
FastAPI 侧的三件事：
    POST /pipeline/run          跑一次完整流水线，返回每环结果（排障主要看它）
    POST /pipeline/tools/index  工具元数据向量入库（新增/更新技能后调用）
    GET  /pipeline/health       看 pgvector 是否可用、库里有多少条工具

【为什么单独一个文件】
    server.py 里所有路由都写在 _register_routes 的闭包中，已经很长了。
    流水线是独立可测的一组能力，单独成文件后既能挂到主 app，
    也能在测试里直接 register_pipeline_routes(FastAPI()) 起一个干净的 app。

【组件为什么惰性创建】
    端点注册发生在应用启动时，此时不能因为 PG 没起来就让整个服务启动失败。
    所以 pipeline / store 都在第一次请求时才创建，创建失败也只是该端点不可用。
"""
from typing import List, Optional  # 类型标注：列表与可选类型

from fastapi import Depends, FastAPI, HTTPException  # FastAPI 核心：依赖注入/应用/异常
from pydantic import BaseModel, Field  # 请求体模型与字段校验

from modules.common.logger import get_logger  # 统一日志器工厂

_logger = get_logger("api.pipeline")  # 本模块日志器，名 "api.pipeline"


# ----------------------------------------------------------------------
# 请求/响应模型
# ----------------------------------------------------------------------
# 【什么】POST /pipeline/run 的请求体
# 【作用】承载用户原始输入，以及调用方额外要排除的工具名
class PipelineRequest(BaseModel):
    """流水线执行请求。"""

    query: str = Field(..., description="用户原始输入")                                              # 必填，空串也收
    extra_exclude: List[str] = Field(default_factory=list, description="额外排除的工具名（如 RBAC 不允许的）")  # 默认不选任何工具
    use_llm_intent: bool = Field(default=True, description="False 时意图识别只走规则（离线/压测用）")     # 默认走 LLM 意图识别


# 【什么】一条待入库的工具元数据
# 【作用】作为 ToolIndexRequest.tools 的元素，描述一个工具的语义与参数
class ToolIndexItem(BaseModel):
    """一条待入库的工具元数据。"""

    name: str                                    # 工具名，入库后按它去重
    description: str = ""                        # 描述文本；语义召回主要靠它，留空则召回不到
    domain_tag: str = "general"                  # 领域标签，默认 general
    source_type: str = "local"                   # 来源：local（本地）/ mcp_remote（MCP 远程）
    server_name: str = ""                        # 所属 MCP Server 名；本地工具留空
    parameters: dict = Field(default_factory=dict)  # 参数表（JSON Schema），默认空对象


# 【什么】POST /pipeline/tools/index 的请求体
# 【作用】一次批量提交多条工具元数据入库
class ToolIndexRequest(BaseModel):
    """批量入库请求。"""

    tools: List[ToolIndexItem]                                                                    # 待入库工具列表
    rebuild_embedding: bool = Field(default=True, description="True 时重新编码向量；False 只更新元数据")  # 默认重新编码向量


# ----------------------------------------------------------------------
# 组件惰性单例
# ----------------------------------------------------------------------
# 【什么】组件惰性单例：流水线对象与向量存储对象
# 【作用】启动时不创建，第一次请求才创建；创建失败保持 None，端点返回 503
_pipeline = None  # 流水线对象：惰性创建，第一次请求才初始化
_store = None     # 向量存储对象：惰性创建，第一次请求才初始化


# 【什么】取（必要时创建）流水线与存储层
# 【作用】把"组件可能没就绪"收敛到一个地方，端点里只判 None
def _get_components():
    """第一次调用时创建流水线与存储层；创建失败返回 (None, None) 而不是抛异常。"""
    global _pipeline, _store  # 声明修改模块级变量
    if _pipeline is None:  # 未初始化：第一次请求时创建
        try:
            from modules.pipeline.tool_store_pg import ToolVectorStore  # 延迟导入：PG 未就绪不影响模块加载
            from modules.pipeline.pipeline import RequestPipeline  # 延迟导入：流水线实现
            from modules.tools.tool_registry import ToolRegistry  # 延迟导入：工具注册器

            _store = ToolVectorStore()  # 创建 PG 向量存储实例
            fallback = ToolRegistry().get_default_tools()  # 取默认工具作为兜底
            _pipeline = RequestPipeline(store=_store, fallback_tools=fallback)  # 创建流水线实例
        except Exception as e:  # 初始化失败（如 PG 未启动）
            _logger.warning(f"流水线组件初始化失败: {e}")  # 记录失败原因
            _pipeline, _store = None, None  # 保持 None，端点会返回 503
    return _pipeline, _store  # 返回（可能为 None）的组件元组


# 【什么】把三个流水线端点挂到 FastAPI app 上
# 【作用】既能挂到主服务，也能在测试里对空 app 单独注册；失败不影响主服务启动
def register_pipeline_routes(app: FastAPI, require_auth: bool = True) -> None:
    """把流水线端点挂到 FastAPI app 上。

    Args:
        app: FastAPI 应用实例
        require_auth: True 时端点需要登录态（生产）；False 时免鉴权（本地调试/单测）
    """
    deps = []  # 依赖列表：默认空，免鉴权
    if require_auth:  # 需要鉴权：加入登录态校验依赖
        from modules.api.auth import get_current_user  # 延迟导入：避免循环依赖
        deps = [Depends(get_current_user)]  # 把鉴权依赖加入列表

    # 【什么】POST /pipeline/run —— 跑一次完整流水线
    # 【作用】返回每环结果与 trace，是排查"为什么选了这个工具"的主要入口
    @app.post("/pipeline/run", dependencies=deps)
    async def pipeline_run(req: PipelineRequest):
        """跑一次完整流水线。

        返回里最值得看的三个字段：
            route.branch              这一轮走哪条支路
            route.tool_calls_allowed  是否允许调工具（置信度<0.5 时为 false）
            trace                     每环耗时与结论，排障时从下往上读
        """
        pipeline, _ = _get_components()  # 取流水线组件（可能为 None）
        if pipeline is None:  # 组件未就绪：返回 503
            raise HTTPException(status_code=503, detail="流水线组件未就绪（检查 PG 与 LLM 配置）")  # 503 服务不可用
        try:
            result = pipeline.run(req.query, extra_exclude=req.extra_exclude or [])  # 执行流水线
            return result.to_dict()  # 返回完整结果（含每环 trace）
        except Exception as e:  # 执行异常
            _logger.error(f"流水线执行失败: {e}")  # 记录错误日志
            raise HTTPException(status_code=500, detail=f"流水线执行失败: {e}")  # 500 内部错误

    # 【什么】POST /pipeline/tools/index —— 工具元数据向量入库
    # 【作用】新增或更新技能后调一次，工具才能被 Tool-RAG 语义召回
    @app.post("/pipeline/tools/index", dependencies=deps)
    async def pipeline_tools_index(req: ToolIndexRequest):
        """工具元数据向量入库：新增或更新技能后调用一次，之后才能被召回。"""
        _, store = _get_components()  # 取向量存储组件（可能为 None）
        if store is None or not store.available:  # 存储不可用：返回 503
            raise HTTPException(status_code=503, detail="pgvector 存储不可用，无法入库")  # 503 服务不可用
        try:
            tools = [t.model_dump() for t in req.tools]  # Pydantic 模型转字典列表
            written = store.upsert_tools(tools)  # 批量写入或更新工具元数据
            return {"written": written, "total": store.count(), "table": store.table}  # 返回写入结果统计
        except Exception as e:  # 写入异常
            raise HTTPException(status_code=500, detail=f"工具入库失败: {e}")  # 500 内部错误

    # 【为什么这个端点不加 deps】它是探活端点，容器探针不会带 token；
    #   与 /health 同理，只暴露"是否可用"，不含业务数据
    # 【什么】GET /pipeline/health —— 流水线探活（不加鉴权）
    # 【作用】只暴露 pgvector 是否可用与工具条数，供容器探针调用
    @app.get("/pipeline/health")
    async def pipeline_health():
        """流水线健康：pgvector 是否可用 + 库里工具条数。"""
        _, store = _get_components()  # 取向量存储组件（可能为 None）
        ok = bool(store and store.available)  # 存储存在且可用
        return {  # 返回探活结果
            "status": "ok" if ok else "degraded",  # 整体状态：ok/降级
            "pgvector": ok,  # pgvector 是否可用
            "table": store.table if store else None,  # 表名；store 为 None 时返回 None
            "tools": store.count() if store else -1,  # 工具条数；store 为 None 时返回 -1
        }
