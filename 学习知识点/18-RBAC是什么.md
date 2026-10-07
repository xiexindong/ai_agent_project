# RBAC 是什么？（一篇搞懂）

> 出处：[vector_store.py#L269](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/rag/vector_store.py#L269)
> 项目实现：[rbac_service.py](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/enterprise/rbac_service.py) / [models.py-Principal](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/enterprise/models.py#L28)

---

## 一句话

**RBAC = Role-Based Access Control，基于角色的访问控制。**

核心做法一句话：**不把权限直接发给人，而是发给"岗位"，人上岗就自动拥有该岗位的权限。**

就像公司发门禁卡：不给"张三"单独配权限，而是给"财务专员"这个角色配好财务室权限；谁担任财务专员，刷卡就能进财务室。张三换岗去了行政，财务权限随角色收回，不用一条条删。

---

## 【为什么要这个？】

**对比例子**：假设直接把权限发给每个人（ACL 方式）

```
公司 1000 人，20 个财务。财务室权限要一个个发给 20 个人；
新来一个财务 → 补发一次；某个财务换岗 → 记得删掉（忘了删就越权）；
财务室新增了档案室权限 → 要给 20 个人每人再发一遍。
```

RBAC 把"人"和"权限"中间加一层"角色"，三处变化都只改一处：

```
        分配                    拥有
用户 ──────────► 角色 ──────────► 权限
张三            财务专员         进财务室
李四            财务专员         查财务库
王五            行政专员         进办公区
```

- 新人来 → 给角色
- 人换岗 → 换角色（旧权限整体带走）
- 岗位新增权限 → 只改角色一次，所有该角色的人立即生效

---

## 【三要素对照表】

| 概念 | 含义 | 本项目的表 | 例子 |
|---|---|---|---|
| 用户 User | 具体的人 | `sys_user` | 张三（U001） |
| 角色 Role | 岗位/身份 | `sys_role` | 财务专员 |
| 权限 Permission | 能做什么 | `sys_permission` | `kb:finance`、`tool:data_query` |
| 用户—角色 | 谁担任什么岗 | `sys_user_role` | U001 → 财务专员 |
| 角色—权限 | 岗位能做什么 | `sys_role_permission` | 财务专员 → kb:finance |

**判断口诀**：权限挂角色，角色挂用户；问"张三能不能查财务库"，顺着 `用户 → 角色 → 权限` 走一遍。

---

## 【本项目说明：一次请求怎么过 RBAC】

入口不是用户名密码，而是一把 **API Key**，解析后拿到 `Principal`（一次请求的权限快照）：

```
请求带 API Key
   │
   ▼
RBACService.resolve(key)     查 4 张表：
   ├─ ① sys_api_key → sys_user        身份：谁、在职吗
   ├─ ② sys_user_role → sys_role      角色 + 数据域 + 脱敏强度
   ├─ ③ sys_role_permission           权限码集合
   └─ ④ sys_kb_grant                  可见知识库 kb_codes
   ▼
Principal（只读快照）= {
   user_code: "U001",
   roles:     ["finance_clerk"],
   permissions: {"tool:data_query", "kb:finance", ...},
   kb_codes:   ["public", "finance"],   ← 只能搜这两个知识库
   data_scope: "dept",                  ← 业务数据只能看本部门
   mask_level: 1,                       ← 返回结果要部分脱敏
}
```

之后整条链路都只看这份快照，不再查库（结果缓存 60 秒）：

| 场景 | 用 Principal 的哪个字段 | 效果 |
|---|---|---|
| 调工具前 | `permissions` / `can()` | 没有 `tool:xxx` 直接拦 |
| 知识库检索（L269 说的） | `kb_codes` | 只在可见库里搜，搜不到无权看的内容 |
| 查业务数据 | `data_scope` | self 只看自己 / dept 看本部门 / all 全公司 |
| 返回结果 | `mask_level` | 手机号、薪资按强度打码 |

回到 L269 那句话：**`search(kb_codes=principal.kb_codes)`** = 直接把"这个人可见哪些知识库"传给检索，系统就不会从他无权看的库里召回内容——权限在"检索阶段"就封死，而不是"搜出来再藏"。

---

## JS 类比

相当于登录后后端返回一个 `user` 对象，前端/后端统一读它判断权限，而不是每个地方写死人名：

```js
// ACL：权限直接发给人 —— 人一多、岗位一变就维护不动
const acl = { zhangsan: ['finance_room', 'finance_db'] }

// RBAC：中间多一层角色
const rolePerms = {
  finance_clerk: ['finance_room', 'kb:finance', 'tool:data_query'],
  admin_clerk:   ['office_room', 'kb:public'],
}
const userRoles = { zhangsan: ['finance_clerk'] }

// 判定"张三能不能做某事"：用户 → 角色 → 权限
function can(user, perm) {
  return userRoles[user]
    .some(role => rolePerms[role].includes(perm))
}
can('zhangsan', 'kb:finance') // true
can('zhangsan', 'kb:hr')      // false
```

本项目的 `Principal.can()`（[models.py#L49](file:///e:/workspace/AI/AI-project/ai_agent_project/modules/enterprise/models.py#L49)）就是这个 `can()` 的 Python 版，还额外支持 `tool:mcp:*` 这类通配。

---

## 30 秒记住

1. RBAC = 权限发给"角色"、角色发给"人"，人随岗走、权限随角色走。
2. 三要素：用户 → 角色 → 权限；两张关联表把它们连起来。
3. 本项目里 API Key 解析成 `Principal` 快照（角色/权限码/可见知识库/数据域/脱敏级别），之后全链路只读它。
4. L269 的意思：把 `principal.kb_codes` 直接传给检索，**先限定库、再召回**，无权内容根本进不了结果。
