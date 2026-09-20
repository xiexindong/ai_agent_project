"""
权限与安全测试：RBAC 判定 + 知识库隔离表达式
============================================
权限这块的原则只有一条：判定不了就拒绝（fail closed）。
越权召回一次就是泄密事故，宁可这次查不出东西。
"""
import pytest

from modules.enterprise.models import Principal
from modules.enterprise.rbac_service import (
    RBACService, _max_scope,
)
from modules.rag.llamaindex_adapter import parse_expr
from modules.common.exceptions import RAGException


def _p(perms=(), kb=(), scope="self") -> Principal:
    return Principal(user_code="U001", permissions=set(perms), kb_codes=list(kb),
                     data_scope=scope)


# ------------------------------------------------------------------
# 【Principal.can】默认拒绝 + `tool:mcp:*` 前缀通配
# ------------------------------------------------------------------
def test_can_exact_match():
    p = _p(["tool:calculator"])
    assert p.can("tool:calculator") is True
    assert p.can("tool:web_search") is False


def test_can_wildcard_prefix():
    p = _p(["tool:mcp:*"])
    assert p.can("tool:mcp:web_search") is True
    assert p.can("tool:calculator") is False        # 通配只覆盖 mcp 前缀


def test_can_denies_when_no_permissions():
    assert _p().can("anything") is False            # 空权限 = 全拒
    assert _p().can("") is False


def test_can_any():
    p = _p(["tool:mcp:web_search"])
    assert p.can_any(["tool:web_search", "tool:mcp:web_search"]) is True
    assert p.can_any(["tool:a", "tool:b"]) is False


def test_can_kb():
    assert _p(kb=["kb_a"]).can_kb("kb_a") is True
    assert _p(kb=["kb_a"]).can_kb("kb_b") is False
    assert _p().can_kb("anything") is True          # 未开启隔离 = 不限制


# ------------------------------------------------------------------
# 【kb_expr】知识库过滤表达式：没开隔离或没配 kb_codes 就返回 None
#   返回 None 会走「不过滤」的旧行为；返回表达式则强制 Milvus 过滤。
# ------------------------------------------------------------------
def _svc(kb_acl: bool = True) -> RBACService:
    # 【为什么用 __new__ 绕过构造】RBACService.__init__ 会读 Settings 并探测 MySQL，
    #   这里只测纯表达式生成逻辑，不该被外部依赖牵连。
    svc = object.__new__(RBACService)
    svc._kb_acl = kb_acl
    return svc


def test_kb_expr_builds_in_clause():
    expr = _svc().kb_expr(_p(kb=["public", "finance"]))
    assert expr == 'kb_code in ["public","finance"]'


def test_kb_expr_none_when_no_codes_or_acl_off():
    assert _svc().kb_expr(_p(kb=[])) is None            # 没配知识库
    assert _svc().kb_expr(None) is None                 # 没登录
    assert _svc(kb_acl=False).kb_expr(_p(kb=["kb_a"])) is None   # 开关关了


# ------------------------------------------------------------------
# 【allowed_tools】模型看不到 = 调不到，从源头杜绝越权
# ------------------------------------------------------------------
class _Tool:
    def __init__(self, name, tool_type="local"):
        self.name = name
        self.tool_type = tool_type


def test_allowed_tools_filters_by_permission():
    svc = _svc()
    tools = [_Tool("calculator"), _Tool("web_search")]
    allowed = svc.allowed_tools(_p(["tool:calculator"]), tools)
    assert [t.name for t in allowed] == ["calculator"]


def test_allowed_tools_mcp_accepts_two_codes():
    svc = _svc()
    tools = [_Tool("web_search", tool_type="mcp")]
    assert svc.allowed_tools(_p(["tool:mcp:*"]), tools) != []      # 认通配码
    assert svc.allowed_tools(_p(["tool:mcp:web_search"]), tools) != []
    assert svc.allowed_tools(_p(["tool:calculator"]), tools) == []  # 没有就不给


def test_allowed_tools_empty_on_anonymous():
    assert _svc().allowed_tools(None, [_Tool("calculator")]) == []  # 未登录绝不给全部


# ------------------------------------------------------------------
# 【_max_scope】多角色数据域冲突时取最大可见范围
# ------------------------------------------------------------------
@pytest.mark.parametrize("scopes,expected", [
    (["self"], "self"),
    (["self", "dept"], "dept"),
    (["dept", "all"], "all"),
    ([], "self"),
    (["unknown_value"], "self"),      # 未知值按 self 处理，不放大
])
def test_max_scope(scopes, expected):
    assert _max_scope(scopes) == expected


# ------------------------------------------------------------------
# 【parse_expr】Milvus 表达式 → 中立过滤结构
#   翻译不了必须抛异常拒绝放行，绝不能退回全量检索。
# ------------------------------------------------------------------
def test_parse_expr_in_clause():
    got = parse_expr('kb_code in ["kb_a","kb_b"]')
    assert got == [{"key": "kb_code", "op": "in", "value": ["kb_a", "kb_b"]}]


def test_parse_expr_multiple_clauses():
    got = parse_expr('level == "public" and kb_code in ["kb_a"]')
    assert got == [
        {"key": "level", "op": "==", "value": "public"},
        {"key": "kb_code", "op": "in", "value": ["kb_a"]},
    ]


def test_parse_expr_empty():
    assert parse_expr("") == [] and parse_expr(None) == [] and parse_expr("   ") == []


def test_parse_expr_rejects_unsupported_operator():
    with pytest.raises(RAGException):
        parse_expr("score > 0.5")        # 比较符不支持 → 拒绝放行


def test_parse_expr_rejects_broken_brackets():
    with pytest.raises(RAGException):
        parse_expr('kb_code in ["kb_a"')   # 缺右括号 → 拒绝放行
