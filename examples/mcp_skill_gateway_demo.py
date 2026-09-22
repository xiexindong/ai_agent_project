"""
MCP 技能网关 + Tool-RAG 动态召回 + 分层 Agent 路由  演示
=======================================================
回答一个问题：远程技能越来越多时，怎么做到「模型每轮只看本轮用得上的那几个」。

【核心思路】
    MCP 远程技能不直连 LLM。
    改造前：启动时把全部 MCP 工具注册进 ToolRegistry → 每轮 ReAct 把全部 schema 发给模型。
    改造后：
        MCP Server ──► 网关（纳管 + 工具目录，不起进程也能索引）
                            │
                            ▼
                       Tool-RAG（工具名+描述 编码成向量）
                            │
             用户提问 ──► 语义召回 top-k ──► L0 路由（技能绑定优先 + 上限截断 + 兜底）
                            │
                            ▼
                  按需连 server + 注册 ──► LLM（只看到 k 个工具）

【七幕】
    第一幕  技能膨胀的账：全量暴露 vs 语义召回，token 差多少（具体数值）
    第二幕  网关纳管：目录来自缓存，一个子进程都不用起
    第三幕  Tool-RAG 召回：同一批工具、不同提问，召回谁、分数多少
    第四幕  分层路由：技能绑定 → 语义召回 → 截断 → 兜底，每一步可见
    第五幕  真连 MCP Server：被召回的工具此刻才起进程（懒连接实测）
    第六幕  故障 server：连不上时目录为空、熔断打开，不影响其余链路
    第七幕  真实 LLM 调用（未配 DEEPSEEK_API_KEY 则跳过）

【实测输出】（python examples/mcp_skill_gateway_demo.py，ZHIPU key 已配 → 走语义召回）
    第一幕  32 个工具（本地 2 + 远程 30）全量 schema = 2631 token / 召回 5 个 = 548 token
            相差 4.8 倍；工具涨到 320 个时前者继续线性增长，后者不变
    第二幕  网关目录 30 个工具，connects=0（目录来自缓存，6 个 server 一个都没起）
    第三幕  '帮我查一下上海的天气' → city_weather=0.5362, get_current_time=0.3386, stock_quote=0.3251
            '我的报销单到哪一步了' → query_reimbursement=0.5656, submit_invoice=0.3794, logistics_track=0.3057
    第四幕  '查一下这个订单的状态' → 工具 5/32：data_query(技能绑定1.0) + 语义召回 4 个
            '现在几点了'           → 工具 2/32：web_search、get_current_time（来源=技能绑定）
            '随便聊聊'             → 工具 2/32：calculator、data_query（来源=默认兜底）
    第五幕  懒连接：真实目录 3 个工具，只注册被召回的 1 个（registered=1），connects=2
    第六幕  坏 server 目录为空 []，熔断 open，第二次拉目录直接跳过连接
    第七幕  LLM 本轮只带 1 个工具：get_current_time（全量模式会带全部）
    注：未配 ZHIPU_API_KEY 时 Tool-RAG 自动降级为关键词召回，流程照跑。
"""
import json
import sys
import tempfile
from pathlib import Path

# demo 在 examples/ 下，直接运行时项目根不在 sys.path 里，先补上
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.common.logger import get_logger
from modules.common.token_counter import count_tokens
from modules.common.circuit_breaker import CircuitBreaker
from modules.mcp.gateway import MCPGateway, MCPServerConfig
from modules.tools.tool_rag import ToolRAGIndex
from modules.tools.tool_registry import ToolRegistry
from modules.agent.router import AgentRouter
from modules.skills import SkillRegistry, get_default_skills

_logger = get_logger("gateway_demo")
QUIET = {"quiet": True}


# ----------------------------------------------------------------------
# 模拟的「海量远程技能」：30 个 MCP 工具，分属 6 个 server。
# 真实项目里这份目录来自 tools/list，这里写死是为了让 demo 可重复运行。
# ----------------------------------------------------------------------
def _demo_catalog() -> list:
    groups = {
        "finance-mcp": [
            ("query_reimbursement", "查询报销单状态与审批进度", {"bill_id": {}, "user_code": {}}),
            ("submit_invoice", "提交发票并校验真伪", {"invoice_code": {}, "amount": {}}),
            ("budget_balance", "查询部门预算余额与使用率", {"dept_code": {}, "year": {}}),
            ("cost_center_report", "生成成本中心月度报表", {"month": {}, "dept_code": {}}),
            ("audit_log_query", "查询财务审计操作日志", {"start_date": {}, "end_date": {}}),
        ],
        "hr-mcp": [
            ("leave_balance", "查询员工年假与调休余额", {"user_code": {}}),
            ("attendance_query", "查询员工考勤打卡记录", {"user_code": {}, "date": {}}),
            ("recruit_pipeline", "查询招聘岗位与候选人进度", {"job_id": {}}),
            ("performance_review", "查询绩效考核结果与评级", {"user_code": {}, "cycle": {}}),
            ("org_chart", "查询组织架构与汇报关系", {"dept_code": {}}),
        ],
        "it-mcp": [
            ("ticket_create", "创建 IT 运维工单", {"title": {}, "priority": {}}),
            ("asset_query", "查询办公资产与领用记录", {"asset_id": {}}),
            ("vpn_reset", "重置员工 VPN 账号密码", {"user_code": {}}),
            ("server_health", "查询服务器健康与负载", {"host": {}}),
            ("permission_grant", "申请系统权限与审批流", {"system": {}, "role": {}}),
        ],
        "data-mcp": [
            ("city_weather", "查询指定城市的实时天气与温度", {"city": {}, "date": {}}),
            ("text_translate", "多语言文本互译", {"text": {}, "target_lang": {}}),
            ("web_search", "网络搜索，获取互联网最新信息与新闻", {"query": {}, "max_results": {}}),
            ("stock_quote", "查询股票实时行情与涨跌幅", {"symbol": {}}),
            ("news_digest", "生成指定主题的新闻摘要", {"topic": {}, "days": {}}),
        ],
        "ops-mcp": [
            ("logistics_track", "查询物流运单轨迹", {"tracking_no": {}}),
            ("warehouse_stock", "查询仓库库存与库位", {"sku": {}, "warehouse": {}}),
            ("delivery_schedule", "安排配送班次与车辆", {"order_id": {}, "date": {}}),
            ("supplier_score", "查询供应商评分与履约率", {"supplier_id": {}}),
            ("container_monitor", "监控集装箱温湿度与位置", {"container_id": {}}),
        ],
        "enterprise-mcp": [
            ("get_current_time", "获取当前时间，支持指定时区", {"timezone": {}}),
            ("knowledge_stats", "获取知识库统计信息", {}),
            ("contract_review", "合同风险条款审查与提示", {"contract_id": {}}),
            ("policy_lookup", "查询公司制度条款原文", {"keyword": {}}),
            ("meeting_room_book", "预订会议室并通知参会人", {"room": {}, "time": {}, "users": {}}),
        ],
    }
    catalog = []
    for server, tools in groups.items():
        for name, desc, props in tools:
            catalog.append({
                "name": name,
                "description": desc,
                "inputSchema": {
                    "type": "object",
                    "properties": {k: {"type": "string", "description": k} for k in props},
                    "required": list(props.keys())[:1],
                },
                "server": server,
            })
    return catalog


def _to_schema(entry: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": entry["name"],
            "description": entry["description"],
            "parameters": entry["inputSchema"],
        },
    }


def act1_bloat_cost(registry: ToolRegistry, catalog: list):
    print("\n" + "=" * 68)
    print("第一幕  技能膨胀的账：全量暴露 vs 语义召回")
    print("=" * 68)
    # 全量 = 本地工具 + 全部远程技能（改造前每轮发的就是这一整份）
    local_schema = [t.to_openai_format() for t in registry.list_tools()]
    all_schema = local_schema + [_to_schema(e) for e in catalog]
    full_tokens = count_tokens(json.dumps(all_schema, ensure_ascii=False))

    # 只暴露 top_k 个（默认 tool_rag_top_k=5）
    small_tokens = count_tokens(json.dumps(local_schema[:2] + [_to_schema(e) for e in catalog[:3]], ensure_ascii=False))

    print(f"工具总数            {len(all_schema)} 个（本地 {len(local_schema)} + 远程 {len(catalog)}）")
    print(f"全量暴露 schema     {full_tokens} token")
    print(f"召回 5 个 schema    {small_tokens} token")
    print(f"相差                {full_tokens / max(1, small_tokens):.1f} 倍")
    print("\n  注：这还只是 tools 参数本身，不含模型在候选变多时的选择成本。")
    print("  工具从 32 涨到 320，全量那一行继续线性增长，召回那一行不变。")


def act2_gateway_catalog(catalog: list, cache_path: Path):
    print("\n" + "=" * 68)
    print("第二幕  MCP 技能网关：目录来自缓存，不起子进程")
    print("=" * 68)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "tools": catalog}, f, ensure_ascii=False)

    servers = [
        MCPServerConfig(name=n, command=[sys.executable, "-m", "modules.mcp.mcp_server"])
        for n in sorted({e["server"] for e in catalog})
    ]
    gw = MCPGateway(
        servers=servers,
        lazy_connect=True,
        catalog_cache=str(cache_path),
        breaker=CircuitBreaker(5, 30),
    )
    tools = gw.catalog()
    print(f"登记的 server       {len(gw.server_names)} 个: {gw.server_names}")
    print(f"工具目录            {len(tools)} 个工具")
    print(f"健康快照            {gw.health()}")
    print("\n  connects=0 是关键：建索引只需要工具描述，不需要把 6 个 server 全拉起来。")
    print("  目录里前 3 个: " + "、".join(t["name"] for t in tools[:3]))
    return cache_path


def act3_recall(catalog: list):
    print("\n" + "=" * 68)
    print("第三幕  Tool-RAG 动态召回：不同提问召回不同工具")
    print("=" * 68)
    idx = ToolRAGIndex(top_k=5, min_score=0.25)
    idx.index_tools(catalog)
    print(f"索引状态            {idx.stats()}")

    for query in ["帮我查一下上海的天气", "我的报销单到哪一步了", "给客户发一封英文邮件"]:
        hits = idx.recall(query)
        detail = ", ".join(f"{h.name}={h.score}({h.source})" for h in hits) or "（无达标结果）"
        print(f"\n  提问: {query}")
        print(f"  召回: {detail}")
    print("\n  min_score=0.25 的作用：宁可召回为空（由路由兜底），也不塞不相关的候选。")
    return idx


def act4_route(idx: ToolRAGIndex, cache_path: Path, registry: ToolRegistry):
    print("\n" + "=" * 68)
    print("第四幕  分层 Agent 路由：技能绑定 → 语义召回 → 截断 → 兜底")
    print("=" * 68)
    # ------------------------------------------------------------------
    # 【这里刻意不装配 server】路由阶段只需要目录做决策；
    #   server 一个都不登记时，ensure_registered 找不到 server 就跳过，
    #   正好演示「决策阶段不加载任何远程技能」。
    # ------------------------------------------------------------------
    gw_offline = MCPGateway(servers=[], catalog_cache=str(cache_path))
    skill_registry = SkillRegistry()
    skill_registry.register_batch(get_default_skills())
    router = AgentRouter(
        skill_registry=skill_registry, tool_index=idx, gateway=gw_offline,
        top_k=5, max_tools=8,
    )
    for query in ["查一下这个订单的状态", "我的报销单到哪一步了", "现在几点了", "随便聊聊"]:
        d = router.route(query, registry=registry)
        print(f"\n  提问: {query}")
        print(f"  决策: {d.describe()}")
        print(f"  理由: {d.reason} | 分数={d.tool_scores}")
    print("\n  截断规则：最多 8 个（max_tools_per_turn）。")
    print("  兜底规则：技能与召回都为空时带 calculator / data_query，不会让模型无工具可用。")
    print("  决策阶段 registered=0：名单定下来之前，一个远程技能都没被加载。")
    print(f"  网关状态 {gw_offline.health()}")
    return router, gw_offline


def act5_lazy_connect(registry: ToolRegistry, tmpdir: Path):
    print("\n" + "=" * 68)
    print("第五幕  真连 MCP Server：被召回的工具此刻才起进程")
    print("=" * 68)
    real = MCPGateway(
        servers=[MCPServerConfig("enterprise-mcp", [sys.executable, "-m", "modules.mcp.mcp_server"])],
        lazy_connect=True,
        catalog_cache=str(tmpdir / "real_catalog.json"),
        connect_retry=2,
        breaker=CircuitBreaker(5, 30),
    )
    try:
        cat = real.catalog()
        print(f"真实目录            {[t['name'] for t in cat]}")
        idx = ToolRAGIndex(top_k=3, min_score=0.25)
        idx.index_tools(cat)
        router = AgentRouter(tool_index=idx, gateway=real, top_k=3, max_tools=4)
        d = router.route("现在几点了", registry=registry)
        print(f"路由决策            {d.describe()}")

        target = next((n for n in d.tool_names if n in ("get_current_time", "knowledge_stats")), None)
        if target:
            res = registry.execute_tool(target, {"timezone": "Asia/Shanghai"}, session_id="demo-lazy")
            print(f"执行 {target}       success={res.success} result={str(res.result)[:60]}")
        print(f"网关健康            {real.health()}")
        print("\n  connects 的含义：第 1 次是拉目录（已有缓存时省掉），第 2 次才是被召回后的按需加载。")
        print("  也就是说：稳定运行时，一轮对话只为真正用到的那 1 个 server 起 1 次进程。")
    finally:
        real.shutdown()


def act6_broken_server(tmpdir: Path):
    print("\n" + "=" * 68)
    print("第六幕  故障 server：目录为空 + 熔断打开，其余链路不受影响")
    print("=" * 68)
    bad = MCPGateway(
        servers=[MCPServerConfig("broken-mcp", [sys.executable, "-m", "modules.not_exist_server_xyz"])],
        lazy_connect=True,
        catalog_cache=str(tmpdir / "broken_catalog.json"),
        connect_retry=2,
        # 阈值设 1 是为了一次失败就看到熔断效果；生产用 CB_FAILURE_THRESHOLD=5
        breaker=CircuitBreaker(1, 30),
    )
    print(f"第 1 次拉目录        {bad.catalog(refresh=True)}（空：连接失败，跳过该 server）")
    print(f"第 2 次拉目录        {bad.catalog(refresh=True)}（熔断已打开，直接跳过连接）")
    print(f"健康快照            {bad.health()}")
    print("\n  关键：坏 server 不影响网关本身可用 —— 目录为空只意味着它贡献 0 个工具，")
    print("  其余 server 的工具照常召回，且不会每次请求都白等一轮连接超时。")


def act7_real_llm(registry: ToolRegistry, idx: ToolRAGIndex, gw: MCPGateway, cache_path: Path = None):
    print("\n" + "=" * 68)
    print("第七幕  真实 LLM 调用（未配 DEEPSEEK_API_KEY 则跳过）")
    print("=" * 68)
    from config import get_settings
    if not get_settings().deepseek_api_key:
        print("  未配置 DEEPSEEK_API_KEY，跳过。")
        return
    try:
        from modules.agent.llm import LLMClient
        skill_registry = SkillRegistry()
        skill_registry.register_batch(get_default_skills())
        router = AgentRouter(skill_registry=skill_registry, tool_index=idx, gateway=gw)
        d = router.route("现在几点了", registry=registry)
        schema = registry.get_openai_tools_schema(allowed=[t for t in registry.list_tools() if t.name in d.tool_names])
        print(f"  本轮只带 {len(schema)} 个工具: {[s['function']['name'] for s in schema]}")
        res = LLMClient().invoke_with_tools(
            messages=[{"role": "user", "content": "现在几点了？"}],
            tools=schema,
        )
        print(f"  LLM 返回工具调用: {res.get('tool_calls')}")
    except Exception as e:
        print(f"  跳过：{e}")


def main():
    print("####################################################################")
    print("#  MCP 技能网关 + Tool-RAG + 分层路由：远程技能不直连 LLM")
    print("####################################################################")

    catalog = _demo_catalog()
    registry = ToolRegistry()
    registry.register_batch(registry.get_default_tools())

    tmpdir = Path(tempfile.mkdtemp(prefix="mcp_gw_demo_"))
    try:
        act1_bloat_cost(registry, catalog)
        cache_path = act2_gateway_catalog(catalog, tmpdir / "demo_catalog.json")
        idx = act3_recall(catalog)
        _router, gw_offline = act4_route(idx, cache_path, registry)
        act5_lazy_connect(registry, tmpdir)
        act6_broken_server(tmpdir)
        act7_real_llm(registry, idx, gw_offline, cache_path)
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("\n" + "=" * 68)
    print("接入清单：这套机制落在哪些文件")
    print("=" * 68)
    print("  modules/mcp/gateway.py     网关：多 server 纳管 / 目录缓存 / 懒连接 / 熔断重试")
    print("  modules/tools/tool_rag.py  Tool-RAG：工具语义索引 + top-k 召回（关键词降级）")
    print("  modules/agent/router.py    L0 路由：技能绑定优先 → 召回 → 截断 → 兜底")
    print("  modules/agent/react_agent.py  reason_step 按名单取工具，不再是全量")
    print("  main.py                    装配三件套；装配失败自动退回全量模式")
    print("\n一句话：网关管'在哪'，Tool-RAG 管'该召回谁'，路由管'最终名单'，模型只看名单上的那几个。")


if __name__ == "__main__":
    main()
