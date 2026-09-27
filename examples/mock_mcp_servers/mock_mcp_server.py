"""
假的多 MCP Server（演示用，不联网、不查库）
=============================================
一个文件假装成「多个业务部门的 MCP Server」：同一份代码，靠 --preset 参数
切换成不同部门的工具集，起 N 个子进程就相当于 N 个独立 MCP Server。

【为什么需要它】
    真实链路里 .env 没配 MCP_SERVERS_JSON，网关只会登记 1 个 enterprise-mcp，
    于是「多 server 纳管」这套机制永远观察不到：
        - 目录来自多个 server 的合并
        - 熔断按 server 名隔离（坏一个不影响其余）
        - 按需加载：只有被召回的工具所属的那个 server 才起进程
    本文件提供 4 个假部门（finance / hr / it / data），让这套机制真正跑起来。

【运行方式】（每个 preset 一个子进程）
    python examples/mock_mcp_servers/mock_mcp_server.py --preset finance
    python examples/mock_mcp_servers/mock_mcp_server.py --preset hr

【协议】JSON-RPC 2.0 + stdio，与 modules/mcp/mcp_server.py 完全一致，
    所以网关 / 客户端不需要任何改动就能连上它。
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

# 子进程以脚本方式启动时 sys.path[0] 是本文件所在目录，补上项目根才能 import modules.*
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.mcp.protocol import (
    parse_message, create_response, create_error,
    METHOD_NOT_FOUND, INTERNAL_ERROR,
)


# ----------------------------------------------------------------------
# 【假工具清单】4 个部门 × 3 个工具 = 12 个
#   每个工具三件套：
#     name        工具名（带部门前缀，避免与其他 server 的工具重名互相覆盖）
#     description 描述，Tool-RAG 靠它做语义召回，写得像人话召回才准
#     reply       假数据实现：接到参数回一段固定文本，不联网不查库
#   工具名重名会怎样：网关 _tool_index 以工具名为 key，后登记的会覆盖先登记的，
#   所以每个部门的工具名都加了前缀。
# ----------------------------------------------------------------------
TOOL_PRESETS = {
    "finance": [
        {
            "name": "finance_reimbursement_query",
            "description": "查询报销单状态与审批进度，返回当前审批节点与预计到账时间",
            "inputSchema": {
                "type": "object",
                "properties": {"bill_id": {"type": "string", "description": "报销单号"}},
                "required": ["bill_id"],
            },
            "reply": lambda a: (
                f"报销单 {a.get('bill_id', '')}：部门经理已审批 → 财务复核中，"
                f"金额 1286.00 元，预计 3 个工作日到账（演示假数据）"
            ),
        },
        {
            "name": "finance_budget_balance",
            "description": "查询部门预算余额与使用率，返回年度预算、已用额度、剩余额度",
            "inputSchema": {
                "type": "object",
                "properties": {"dept_code": {"type": "string", "description": "部门编码"}},
                "required": ["dept_code"],
            },
            "reply": lambda a: (
                f"部门 {a.get('dept_code', '')} 年度预算 500,000 元，已用 312,450 元，"
                f"剩余 187,550 元，使用率 62.5%（演示假数据）"
            ),
        },
        {
            "name": "finance_invoice_submit",
            "description": "提交发票并校验真伪，返回校验结果与发票明细",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "invoice_code": {"type": "string", "description": "发票代码"},
                    "amount": {"type": "string", "description": "开票金额"},
                },
                "required": ["invoice_code"],
            },
            "reply": lambda a: (
                f"发票 {a.get('invoice_code', '')} 校验通过，金额 {a.get('amount', '未填')} 元，"
                f"已入账待复核（演示假数据）"
            ),
        },
    ],
    "hr": [
        {
            "name": "hr_leave_balance",
            "description": "查询员工年假与调休余额，返回可用天数与过期时间",
            "inputSchema": {
                "type": "object",
                "properties": {"user_code": {"type": "string", "description": "员工工号"}},
                "required": ["user_code"],
            },
            "reply": lambda a: (
                f"员工 {a.get('user_code', '')}：年假剩余 7.5 天（2026-12-31 过期），"
                f"调休剩余 12 小时（演示假数据）"
            ),
        },
        {
            "name": "hr_attendance_query",
            "description": "查询员工考勤打卡记录，返回当日上下班打卡时间",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "user_code": {"type": "string", "description": "员工工号"},
                    "date": {"type": "string", "description": "日期，如 2026-09-23"},
                },
                "required": ["user_code"],
            },
            "reply": lambda a: (
                f"员工 {a.get('user_code', '')} 在 {a.get('date', '今天')} 打卡："
                f"上班 09:02，下班 18:41，无异常（演示假数据）"
            ),
        },
        {
            "name": "hr_org_chart",
            "description": "查询组织架构与汇报关系，返回部门负责人与上级部门",
            "inputSchema": {
                "type": "object",
                "properties": {"dept_code": {"type": "string", "description": "部门编码"}},
                "required": ["dept_code"],
            },
            "reply": lambda a: (
                f"部门 {a.get('dept_code', '')}：负责人 张伟（工号 E1002），"
                f"上级部门 研发中心，下级 3 个小组（演示假数据）"
            ),
        },
    ],
    "it": [
        {
            "name": "it_ticket_create",
            "description": "创建 IT 运维工单，返回工单号与受理人",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "工单标题"},
                    "priority": {"type": "string", "description": "优先级：高/中/低"},
                },
                "required": ["title"],
            },
            "reply": lambda a: (
                f"工单已创建：IT-20260923-0187，标题「{a.get('title', '')}」，"
                f"优先级 {a.get('priority', '中')}，受理人 IT 值班组（演示假数据）"
            ),
        },
        {
            "name": "it_asset_query",
            "description": "查询办公资产与领用记录，返回资产型号与领用人",
            "inputSchema": {
                "type": "object",
                "properties": {"asset_id": {"type": "string", "description": "资产编号"}},
                "required": ["asset_id"],
            },
            "reply": lambda a: (
                f"资产 {a.get('asset_id', '')}：ThinkPad T14，领用人 李娜（E2051），"
                f"领用日期 2025-03-11（演示假数据）"
            ),
        },
        {
            "name": "it_server_health",
            "description": "查询服务器健康与负载，返回 CPU、内存、磁盘使用率",
            "inputSchema": {
                "type": "object",
                "properties": {"host": {"type": "string", "description": "主机名或 IP"}},
                "required": ["host"],
            },
            "reply": lambda a: (
                f"主机 {a.get('host', '')}：CPU 43%，内存 61%，磁盘 78%，"
                f"负载 1.24，状态正常（演示假数据）"
            ),
        },
    ],
    "data": [
        {
            "name": "data_city_weather",
            "description": "查询指定城市的实时天气与温度，返回天气状况与气温",
            "inputSchema": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名"}},
                "required": ["city"],
            },
            "reply": lambda a: (
                f"{a.get('city', '')}：多云转晴，气温 26℃，东南风 3 级，"
                f"湿度 58%（演示假数据）"
            ),
        },
        {
            "name": "data_stock_quote",
            "description": "查询股票实时行情与涨跌幅，返回最新价与涨跌百分比",
            "inputSchema": {
                "type": "object",
                "properties": {"symbol": {"type": "string", "description": "股票代码"}},
                "required": ["symbol"],
            },
            "reply": lambda a: (
                f"{a.get('symbol', '')}：最新价 128.66，涨跌 +2.31（+1.83%），"
                f"成交额 12.4 亿（演示假数据）"
            ),
        },
        {
            "name": "data_text_translate",
            "description": "多语言文本互译，返回翻译后的文本",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "待翻译文本"},
                    "target_lang": {"type": "string", "description": "目标语言，如 en / ja"},
                },
                "required": ["text"],
            },
            "reply": lambda a: (
                f"[{a.get('target_lang', 'en')}] "
                f"Translated: {a.get('text', '')}（演示假数据，未真实调用翻译服务）"
            ),
        },
    ],
}


# 【什么】假 MCP Server：按 preset 暴露不同工具集，返回写死的假数据
# 【作用】一个文件顶 N 个部门 server，用来让多 server 纳管机制跑起来
class MockMCPServer:
    """按 --preset 切换工具集的假 MCP Server。"""

    def __init__(self, preset: str):
        self._preset = preset
        self._tools = TOOL_PRESETS.get(preset, [])

    # 【什么】请求分发，与真实 mcp_server.py 完全同构
    # 【作用】initialize / tools/list / tools/call 三种 method，其余回 -32601
    def handle_request(self, msg: dict) -> Optional[str]:
        req_id = msg.get("id", 0)
        method = msg.get("method", "")
        try:
            if method == "initialize":
                return create_response(req_id, {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": f"mock-{self._preset}-mcp", "version": "1.0.0"},
                })
            if method == "notifications/initialized":
                return None
            if method == "tools/list":
                return create_response(req_id, {
                    "tools": [{k: v for k, v in t.items() if k != "reply"} for t in self._tools],
                })
            if method == "tools/call":
                return self._handle_tool_call(msg, req_id)
            return create_error(req_id, METHOD_NOT_FOUND, f"未知方法: {method}")
        except Exception as e:
            return create_error(req_id, INTERNAL_ERROR, str(e))

    # 【什么】执行一个假工具：找到定义 → 调 reply → 包成 MCP 的文本内容
    # 【作用】工具不存在回 -32601；结果格式与真实 MCP Server 一致
    def _handle_tool_call(self, msg: dict, req_id: int) -> str:
        params = msg.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        for t in self._tools:
            if t["name"] == tool_name:
                text = t["reply"](arguments)
                return create_response(req_id, {
                    "content": [{"type": "text", "text": str(text)}],
                    "isError": False,
                })
        return create_error(req_id, METHOD_NOT_FOUND, f"工具不存在: {tool_name}")

    # 【什么】stdio 事件循环：一行请求 → 一行响应
    # 【作用】stdout 只写协议行（日志走 stderr），否则协议会被日志冲垮
    def run(self):
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                resp = self.handle_request(parse_message(line))
                if resp:
                    print(resp, flush=True)
            except Exception:
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="假的部门 MCP Server")
    parser.add_argument(
        "--preset", default="finance",
        choices=sorted(TOOL_PRESETS.keys()),
        help="部门预设，决定本进程暴露哪一套工具",
    )
    MockMCPServer(parser.parse_args().preset).run()
