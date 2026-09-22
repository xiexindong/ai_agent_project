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
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from modules.common.logger import get_logger

_logger = get_logger("api.pipeline")


# ----------------------------------------------------------------------
# 请求/响应模型
# ----------------------------------------------------------------------
class PipelineRequest(BaseModel):
    """流水线执行请求。"""

    query: str = Field(..., description="用户原始输入")
    extra_exclude: List[str] = Field(default_factory=list, description="额外排除的工具名（如 RBAC 不允许的）")
    use_llm_intent: bool = Field(default=True, description="False 时意图识别只走规则（离线/压测用）")


class ToolIndexItem(BaseModel):
    """一条待入库的工具元数据。"""

    name: str
    description: str = ""
    domain_tag: str = "general"
    source_type: str = "local"          # local / mcp_remote
    server_name: str = ""
    parameters: dict = Field(default_factory=dict)


class ToolIndexRequest(BaseModel):
    """批量入库请求。"""

    tools: List[ToolIndexItem]
    rebuild_embedding: bool = Field(default=True, description="True 时重新编码向量；False 只更新元数据")


# ----------------------------------------------------------------------
# 组件惰性单例
# ----------------------------------------------------------------------
_pipeline = None
_store = None


def _get_components():
    """第一次调用时创建流水线与存储层；创建失败返回 (None, None) 而不是抛异常。"""
    global _pipeline, _store
    if _pipeline is None:
        try:
            from modules.pipeline.tool_store_pg import ToolVectorStore
            from modules.pipeline.pipeline import RequestPipeline
            from modules.tools.tool_registry import ToolRegistry

            _store = ToolVectorStore()
            fallback = ToolRegistry().get_default_tools()
            _pipeline = RequestPipeline(store=_store, fallback_tools=fallback)
        except Exception as e:
            _logger.warning(f"流水线组件初始化失败: {e}")
            _pipeline, _store = None, None
    return _pipeline, _store


def register_pipeline_routes(app: FastAPI, require_auth: bool = True) -> None:
    """把流水线端点挂到 FastAPI app 上。

    Args:
        app: FastAPI 应用实例
        require_auth: True 时端点需要登录态（生产）；False 时免鉴权（本地调试/单测）
    """
    deps = []
    if require_auth:
        from modules.api.auth import get_current_user
        deps = [Depends(get_current_user)]

    @app.post("/pipeline/run", dependencies=deps)
    async def pipeline_run(req: PipelineRequest):
        """跑一次完整流水线。

        返回里最值得看的三个字段：
            route.branch              这一轮走哪条支路
            route.tool_calls_allowed  是否允许调工具（置信度<0.5 时为 false）
            trace                     每环耗时与结论，排障时从下往上读
        """
        pipeline, _ = _get_components()
        if pipeline is None:
            raise HTTPException(status_code=503, detail="流水线组件未就绪（检查 PG 与 LLM 配置）")
        try:
            result = pipeline.run(req.query, extra_exclude=req.extra_exclude or [])
            return result.to_dict()
        except Exception as e:
            _logger.error(f"流水线执行失败: {e}")
            raise HTTPException(status_code=500, detail=f"流水线执行失败: {e}")

    @app.post("/pipeline/tools/index", dependencies=deps)
    async def pipeline_tools_index(req: ToolIndexRequest):
        """工具元数据向量入库：新增或更新技能后调用一次，之后才能被召回。"""
        _, store = _get_components()
        if store is None or not store.available:
            raise HTTPException(status_code=503, detail="pgvector 存储不可用，无法入库")
        try:
            tools = [t.model_dump() for t in req.tools]
            written = store.upsert_tools(tools)
            return {"written": written, "total": store.count(), "table": store.table}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"工具入库失败: {e}")

    # 【为什么这个端点不加 deps】它是探活端点，容器探针不会带 token；
    #   与 /health 同理，只暴露"是否可用"，不含业务数据
    @app.get("/pipeline/health")
    async def pipeline_health():
        """流水线健康：pgvector 是否可用 + 库里工具条数。"""
        _, store = _get_components()
        ok = bool(store and store.available)
        return {
            "status": "ok" if ok else "degraded",
            "pgvector": ok,
            "table": store.table if store else None,
            "tools": store.count() if store else -1,
        }
