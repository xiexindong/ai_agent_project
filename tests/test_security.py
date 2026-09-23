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


# 【什么】造测试用 Principal 的辅助函数
# 【作用】只填权限码 / 可见库 / 数据域三项，其余字段走模型默认值
def _p(perms=(), kb=(), scope="self") -> Principal:
    return Principal(user_code="U001", permissions=set(perms), kb_codes=list(kb),
                     data_scope=scope)


# ------------------------------------------------------------------
# 【Principal.can】默认拒绝 + `tool:mcp:*` 前缀通配
# ------------------------------------------------------------------
# 【什么】精确匹配权限码的用例
# 【作用】验：有 tool:calculator 时通过，没有 tool:web_search 时拒绝
def test_can_exact_match():
    p = _p(["tool:calculator"])
    assert p.can("tool:calculator") is True
    assert p.can("tool:web_search") is False


# 【什么】tool:mcp:* 前缀通配的用例
# 【作用】验：通配能覆盖 tool:mcp:web_search，但覆盖不到 tool:calculator
def test_can_wildcard_prefix():
    p = _p(["tool:mcp:*"])
    assert p.can("tool:mcp:web_search") is True
    assert p.can("tool:calculator") is False        # 通配只覆盖 mcp 前缀


# 【什么】空权限一律拒绝的用例
# 【作用】验：没有任何权限码时，任意权限名与空串都判 False
def test_can_denies_when_no_permissions():
    assert _p().can("anything") is False            # 空权限 = 全拒
    assert _p().can("") is False


# 【什么】多码取或（can_any）的用例
# 【作用】验：候选里有一个命中就 True，一个都不命中才 False
def test_can_any():
    p = _p(["tool:mcp:web_search"])
    assert p.can_any(["tool:web_search", "tool:mcp:web_search"]) is True
    assert p.can_any(["tool:a", "tool:b"]) is False


# 【什么】知识库可见性判定的用例
# 【作用】验：kb_codes 为空表示没开隔离，此时一律放行
def test_can_kb():
    assert _p(kb=["kb_a"]).can_kb("kb_a") is True
    assert _p(kb=["kb_a"]).can_kb("kb_b") is False
    assert _p().can_kb("anything") is True          # 未开启隔离 = 不限制


# ------------------------------------------------------------------
# 【kb_expr】知识库过滤表达式：没开隔离或没配 kb_codes 就返回 None
#   返回 None 会走「不过滤」的旧行为；返回表达式则强制 Milvus 过滤。
# ------------------------------------------------------------------
# 【什么】造一个只有 kb_acl 开关的 RBACService
# 【作用】绕开要读配置、要探测 MySQL 的构造过程，只测纯表达式逻辑
def _svc(kb_acl: bool = True) -> RBACService:
    # 【为什么用 __new__ 绕过构造】RBACService.__init__ 会读 Settings 并探测 MySQL，
    #   这里只测纯表达式生成逻辑，不该被外部依赖牵连。
    svc = object.__new__(RBACService)
    svc._kb_acl = kb_acl
    return svc


# 【什么】生成 in 过滤表达式的用例
# 【作用】验：可见库为 public/finance 时，表达式逐字等于 kb_code in [...]
def test_kb_expr_builds_in_clause():
    expr = _svc().kb_expr(_p(kb=["public", "finance"]))
    assert expr == 'kb_code in ["public","finance"]'


# 【什么】三种不该生成表达式的情形
# 【作用】验：没配库 / 没登录 / 隔离开关关闭，三种都返回 None
def test_kb_expr_none_when_no_codes_or_acl_off():
    assert _svc().kb_expr(_p(kb=[])) is None            # 没配知识库
    assert _svc().kb_expr(None) is None                 # 没登录
    assert _svc(kb_acl=False).kb_expr(_p(kb=["kb_a"])) is None   # 开关关了


# ------------------------------------------------------------------
# 【allowed_tools】模型看不到 = 调不到，从源头杜绝越权
# ------------------------------------------------------------------
# 【什么】测试用的假工具对象
# 【作用】只带 name 与 tool_type 两个字段，够 allowed_tools 判定即可
class _Tool:
    def __init__(self, name, tool_type="local"):
        self.name = name                     # 工具名，拼成 tool:{name} 去比对
        self.tool_type = tool_type           # local / mcp，mcp 会多认一个权限码


# 【什么】按权限过滤工具列表的用例
# 【作用】验：只有 calculator 通过，web_search 被从名单里剔除
def test_allowed_tools_filters_by_permission():
    svc = _svc()
    tools = [_Tool("calculator"), _Tool("web_search")]
    allowed = svc.allowed_tools(_p(["tool:calculator"]), tools)
    assert [t.name for t in allowed] == ["calculator"]


# 【什么】MCP 工具认两种权限码的用例
# 【作用】验：通配码 tool:mcp:* 与精确码 tool:mcp:web_search 都放行
def test_allowed_tools_mcp_accepts_two_codes():
    svc = _svc()
    tools = [_Tool("web_search", tool_type="mcp")]
    assert svc.allowed_tools(_p(["tool:mcp:*"]), tools) != []      # 认通配码
    assert svc.allowed_tools(_p(["tool:mcp:web_search"]), tools) != []
    assert svc.allowed_tools(_p(["tool:calculator"]), tools) == []  # 没有就不给


# 【什么】未登录身份的用例
# 【作用】验：principal 为 None 时返回空列表，而不是返回全部工具
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
# 【什么】多角色数据域冲突的参数化用例
# 【作用】验：取最宽的那个域；空列表与未知值都落到最窄的 self
def test_max_scope(scopes, expected):
    assert _max_scope(scopes) == expected


# ------------------------------------------------------------------
# 【parse_expr】Milvus 表达式 → 中立过滤结构
#   翻译不了必须抛异常拒绝放行，绝不能退回全量检索。
# ------------------------------------------------------------------
# 【什么】单个 in 子句的解析用例
# 【作用】验：表达式被翻成一条 {key, op, value} 的中立结构
def test_parse_expr_in_clause():
    got = parse_expr('kb_code in ["kb_a","kb_b"]')
    assert got == [{"key": "kb_code", "op": "in", "value": ["kb_a", "kb_b"]}]


# 【什么】and 连接多个子句的解析用例
# 【作用】验：level == 与 kb_code in 被拆成按原顺序排列的两条
def test_parse_expr_multiple_clauses():
    got = parse_expr('level == "public" and kb_code in ["kb_a"]')
    assert got == [
        {"key": "level", "op": "==", "value": "public"},
        {"key": "kb_code", "op": "in", "value": ["kb_a"]},
    ]


# 【什么】空输入与空白输入的用例
# 【作用】验：空串 / None / 纯空格都得到空列表，而不是报错
def test_parse_expr_empty():
    assert parse_expr("") == [] and parse_expr(None) == [] and parse_expr("   ") == []


# 【什么】遇到不支持的比较符的用例
# 【作用】验：score > 0.5 抛 RAGException，拒绝退回全量检索
def test_parse_expr_rejects_unsupported_operator():
    with pytest.raises(RAGException):
        parse_expr("score > 0.5")        # 比较符不支持 → 拒绝放行


# 【什么】括号不闭合的用例
# 【作用】验：畸形表达式抛 RAGException，不做半截解析
def test_parse_expr_rejects_broken_brackets():
    with pytest.raises(RAGException):
        parse_expr('kb_code in ["kb_a"')   # 缺右括号 → 拒绝放行
