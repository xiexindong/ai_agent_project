# 面试14：为什么用 @handle_exception 统一兜异常，而不是各处写 try/except？

> 配套代码：`modules/common/exceptions.py:83-119`（装饰器本体）、使用示例遍布 `modules/memory/pg_manager.py`（每个数据库方法一个）
> 一句话结论：**@handle_exception 把"记日志 + 给兜底值 + 决定要不要继续抛"三件事从每个方法里抽出来变成一行声明——读方法要兜底值（宁可没有记忆也不中断对话），写方法要 reraise（记忆写丢了必须让上层知道并降级），同一个装饰器用参数切换两种策略。**

---

## 一、先说结论

| 面试问 | 一句话答 |
|---|---|
| 解决什么问题？ | 同一段 try/except 样板在几十个方法里复制粘贴：口径不一（有的吞异常、有的抛、日志格式各异） |
| 三个参数各管什么？ | `default` 异常时返回什么 / `log_msg` 日志前缀 / `reraise` 记完日志后吞掉还是继续抛 |
| 读和写的用法差异？ | 读：`default=[]`（召回失败当没记忆，对话继续）；写：`reraise=True`（落库失败必须上抛降级） |
| 异常怎么分类？ | `AgentException`（业务异常，带 message/detail）走一套分支；其它 `Exception` 走另一套 |

---

## 二、大白话拆解

```python
# exceptions.py:83 签名
def handle_exception(
    default: Any = None,           # 异常时返回的兜底值
    log_msg: str = "操作异常",      # 日志前缀（日志里一眼定位是哪个环节）
    reraise: bool = False,         # True = 记完日志继续向上抛；False = 吞掉返回 default
) -> Callable:
```

| 部分 | 大白话解释 |
|---|---|
| `@handle_exception(...)` | 给函数穿一件"异常防护服"：里面炸了不往外溅，按你预设的方式处理 |
| `default=[]` | 炸了就当"查到空列表"——调用方拿到的还是 list，不用判类型 |
| `default=None` | 炸了返回 None——调用方本来就判 None（如 get_session_state 没记录就返回 {}） |
| `reraise=True` | 日志照记，但异常继续飞——给上层的信号："这次失败了，你该走降级路径" |
| 两个 `@overload` | 类型提示重载：让 mypy 知道 reraise=True 时返回类型不含 None |

### 两种典型用法（pg_manager.py 实例）

```python
# 读方法：吞异常 + 兜底值
@handle_exception(default=[], log_msg="读取短期记忆异常")
def get_short_term_history(self, ...) -> list: ...
#   数据库抖一下 → 返回 [] → 对话少一段上下文，但绝不中断

# 写方法：吞掉就没有失败信号了，必须上抛
@handle_exception(default=None, log_msg="保存对话记忆异常", reraise=True)
def save_history(self, ...) -> int | None: ...
#   落库失败 → 记日志 → 继续抛 → 上层 agent 决定降级（如跳过记忆继续回答）
```

---

## 三、输入 → 输出对照

```python
# 场景1：读方法吞异常（default=[]）
# 输入: 数据库连接超时，get_short_term_history 被调用
# 内部: except 分支记日志 "读取短期记忆异常: connection timeout"
# 输出: []            ← 调用方 build_context 拿 []，继续拼 prompt，对话不中断

# 场景2：写方法上抛（reraise=True）
# 输入: save_history 落库时表锁超时
# 内部: 记日志 "保存对话记忆异常: lock timeout" → raise
# 输出: 异常向上飞     ← 上层 save 节点捕获并降级（本轮回答照出，记忆这轮丢了）

# 场景3：非 AgentException 的普通异常
# 输入: 代码 bug 抛 KeyError
# 处理: 包装成 AgentException(带 message/detail) 再抛（reraise=True 时）
#       或直接返回 default（reraise=False 时）—— 两种路径日志格式一致
```

---

## 四、为什么要这么写

| 做法 | 原因（不这么写会怎样） |
|---|---|
| 装饰器抽样板 | 几十个 DB 方法各写一遍 try/except：有人吞、有人抛、日志格式五花八门——出问题时日志对不上口径 |
| 读/写策略用参数区分而不是写两个装饰器 | 本质是同一套流程（记日志→兜底/上抛），只是决策点不同；两个装饰器 = 两份要同步维护的代码 |
| `log_msg` 按方法定制（"读取短期记忆异常"） | 统一写"操作异常"：日志里全是同一句话，排障时定位不到是哪个方法炸的 |
| 业务异常与普通异常分开分支 | AgentException 自带 message/detail 结构化信息，直接打；普通异常只有类型+文本，包装后再打保证日志字段一致 |
| 兜底值类型与方法返回类型一致（读 list 给 []） | 给 None 的话调用方 `for r in results` 直接 TypeError——兜底值本身不能成为新崩溃源 |

---

## 五、等价写法对照（展开版）

```python
# 装饰器写法（项目现状，一行声明）
@handle_exception(default=[], log_msg="读取短期记忆异常")
def get_short_term_history(self, ...): ...

# 等价的手写展开
def get_short_term_history(self, ...):
    try:
        ...原逻辑...
    except AgentException as e:
        _logger.error(f"读取短期记忆异常: {e.message} {e.detail}")
        return []
    except Exception as e:
        _logger.error(f"读取短期记忆异常: {type(e).__name__}: {e}")
        return []
#   同样 6 行样板 × 几十个方法 = 几百行纯重复；装饰器把重复归零
```

---

## 六、简单总结

1. 一个装饰器同时做完三件事：**记日志**（log_msg 定位环节）、**给兜底**（default 与返回类型一致）、**定去留**（reraise 切换吞/抛）
2. 读/写分界线：读方法 `default=[]`/`None`（**宁可没有也不中断对话**），写方法 `reraise=True`（**失败必须让上层降级**，静默吞掉等于丢失失败信号）
3. 最大的坑是兜底值类型不一致：返回 list 的方法兜底必须给 `[]`，给 None 会让调用方的 for 循环当场 TypeError——兜底自己变成新崩溃源
