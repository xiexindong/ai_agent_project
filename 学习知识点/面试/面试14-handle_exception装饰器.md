# 面试14：为什么用 @handle_exception 统一兜异常，而不是各处写 try/except？

> 配套代码：`modules/common/exceptions.py:83-121`（装饰器本体）、`exceptions.py:21-28`（AgentException 基类）、使用示例遍布 `modules/memory/pg_manager.py`（每个数据库方法一个，如 pg_manager.py:565 / 654）与 `modules/memory/memory_manager.py:217`
> 一句话结论：**@handle_exception 把"记日志 + 给兜底值 + 决定要不要继续抛"三件事从每个方法里抽出来变成一行声明——读方法要兜底值（宁可没有记忆也不中断对话），写方法要 reraise（记忆写丢了必须让上层知道并降级），同一个装饰器用参数切换两种策略。**

---

## 〇、先把术语说人话

| 术语 | 大白话解释（生活比喻） |
|---|---|
| 装饰器（Decorator） | 给函数**穿一件外套**：函数本体不用改，外套在"穿上"那一刻自动获得新能力（这里是异常防护）。写法就是在函数头顶加一行 `@xxx` |
| 兜底值（default） | 出事时**替真身上场的替身**：函数正常该返回什么类型，替身就装成什么类型，调用方完全感知不到里面炸过 |
| reraise（重新抛出） | "我记下了，但这事**不能瞒着领导**"——记完日志后把异常继续往上递，让上层决定怎么办 |
| 吞掉异常 | "我记下了，这事**到此为止**"——调用方拿到兜底值，程序照常往下走 |
| AgentException | 项目自定义的异常基类（exceptions.py:21-28），像**标准格式的故障报告单**：固定带 message（给用户看的）和 detail（给排查看的）两栏 |
| @overload | 给类型检查器看的**多份说明书**：同一个函数按参数不同有两种用法，各写一份类型声明，让 mypy 检查得更准，运行时不生效 |

---

## 一、先说结论

| 面试问 | 一句话答 |
|---|---|
| 解决什么问题？ | 同一段 try/except 样板在几十个方法里复制粘贴：口径不一（有的吞异常、有的抛、日志格式各异） |
| 三个参数各管什么？ | `default` 异常时返回什么 / `log_msg` 日志前缀 / `reraise` 记完日志后吞掉还是继续抛 |
| 读和写的用法差异？ | 读：`default=[]`（召回失败当没记忆，对话继续）；写：`reraise=True`（落库失败必须上抛降级） |
| 异常怎么分类？ | `AgentException`（业务异常，带 message/detail）走一套分支（exceptions.py:106-113）；其它 `Exception` 走另一套（exceptions.py:114-119） |

---

## 二、大白话拆解

```python
# exceptions.py:83-87 签名
def handle_exception(
    default: Any = None,           # 异常时返回的兜底值
    log_msg: str = "操作异常",      # 日志前缀（日志里一眼定位是哪个环节）
    reraise: bool = False,         # True = 记完日志继续向上抛；False = 吞掉返回 default
) -> Callable:
```

装饰器本体（exceptions.py:98-120）的核心结构：

```python
def decorator(func):
    @functools.wraps(func)          # 保留原函数的 __name__ / __doc__
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)                # 正常路径：原样执行
        except AgentException as e:                     # 分支一：业务异常，有 message/detail
            _logger.error(f"{log_msg} | {type(e).__name__}: {e.message} | detail={e.detail}")
            if reraise:
                raise                                   # 原样抛，保留类型与堆栈
            return default
        except Exception as e:                          # 分支二：未知异常（如 KeyError）
            _logger.error(f"{log_msg} | 未知异常: {type(e).__name__}: {str(e)}")
            if reraise:
                raise AgentException(log_msg, str(e))   # 包装成标准格式再抛
            return default
    return wrapper
```

| 部分 | 大白话解释 |
|---|---|
| `@handle_exception(...)` | 给函数穿一件"异常防护服"：里面炸了不往外溅，按你预设的方式处理 |
| `default=[]` | 炸了就当"查到空列表"——调用方拿到的还是 list，不用判类型 |
| `default=None` | 炸了返回 None——调用方本来就判 None（如 save_history 返回 `int \| None`） |
| `reraise=True` | 日志照记，但异常继续飞——给上层的信号："这次失败了，你该走降级路径" |
| 两个 `@overload`（exceptions.py:63-80） | 类型提示重载：让 mypy 知道 reraise=False 时返回值类型原样保留 |
| `@functools.wraps`（exceptions.py:100） | 保留原函数的 `__name__`/`__doc__`，否则日志和调试里所有函数都叫 wrapper |

### 两种典型用法（真实代码实例）

```python
# 读方法：吞异常 + 兜底值 —— pg_manager.py:653-655
@handle_exception(default=[], log_msg="读取短期记忆异常")
def get_short_term_history(self, session_id: str, user_id: str, rounds: int = 3) -> list: ...
#   数据库抖一下 → 返回 [] → 对话少一段上下文，但绝不中断

# 同样的读兜底思路 —— memory_manager.py:217
@handle_exception(default=[], log_msg="长期记忆召回异常")
def _recall_long_term(self, query: str, user_id: str, ...) -> list: ...

# 写方法：吞掉就没有失败信号了，必须上抛 —— pg_manager.py:565-566
@handle_exception(default=None, log_msg="保存对话记忆异常", reraise=True)
def save_history(self, ...) -> int | None: ...
#   落库失败 → 记日志 → 继续抛 → 上层 agent 决定降级（如跳过记忆继续回答）
```

---

## 三、执行流程图

```
调用方 → wrapper（exceptions.py:103）
              │
              ▼
        try: func(...) ── 正常 ──→ 原样返回结果
              │
              ▼ 炸了，看异常类型
     ┌────────────────────┬────────────────────────┐
     │ AgentException      │ 其它 Exception（如 KeyError）│
     │ （带 message/detail）│ （只有类型名+文本）          │
     ▼                    ▼
  记日志（标准格式）        记日志（包装后格式）
     │                    │
     ▼                    ▼
 reraise=True？        reraise=True？
   ├─ 是 → 原样 raise     ├─ 是 → raise AgentException(log_msg, str(e))
   └─ 否 → return default └─ 否 → return default
```

---

## 四、输入 → 处理 → 输出对照

| 场景 | 输入（出了什么事） | 内部处理 | 输出（调用方拿到什么） |
|---|---|---|---|
| 读方法吞异常 | `get_short_term_history` 执行时数据库连接超时 | except 分支记日志 `"读取短期记忆异常 \| 未知异常: OperationalError: ..."` | `[]` —— build_context 拿空列表继续拼 prompt，对话不中断 |
| 写方法上抛 | `save_history` 落库时表锁超时 | 记日志 `"保存对话记忆异常 \| ..."` → `raise AgentException(...)` | 异常向上飞 —— 上层 save 节点捕获并降级（本轮回答照出，记忆这轮丢了） |
| 未知异常兜底 | 代码 bug 在 `list_user_sessions` 里抛 `KeyError` | 走分支二，记 `"未知异常: KeyError: ..."`，reraise=False | `default=[]`，日志里保留了异常类型名，排查时能定位 |

---

## 五、为什么要这么写

| 做法 | 原因（不这么写会怎样） |
|---|---|
| 装饰器抽样板 | 几十个 DB 方法各写一遍 try/except：有人吞、有人抛、日志格式五花八门——出问题时日志对不上口径。项目里 pg_manager.py 一个文件就有 30+ 个 `@handle_exception` |
| 读/写策略用参数区分而不是写两个装饰器 | 本质是同一套流程（记日志→兜底/上抛），只是决策点不同；两个装饰器 = 两份要同步维护的代码 |
| `log_msg` 按方法定制（"读取短期记忆异常"） | 统一写"操作异常"：日志里全是同一句话，排障时定位不到是哪个方法炸的 |
| 业务异常与普通异常分开分支（exceptions.py:106-119） | AgentException 自带 message/detail 结构化信息，直接打；普通异常只有类型+文本，包装后再打保证日志字段一致 |
| 兜底值类型与方法返回类型一致（读 list 给 []） | 给 None 的话调用方 `for r in results` 直接 TypeError——兜底值本身不能成为新崩溃源 |
| reraise 时未知异常包装成 AgentException（exceptions.py:118） | 上层只需 catch 一类异常就能覆盖所有失败；裸 KeyError 飞上去，上层要写 `except Exception` 这种大网才能接住 |

---

## 六、等价写法对照（展开版）

```python
# 装饰器写法（项目现状，一行声明）—— pg_manager.py:653-655
@handle_exception(default=[], log_msg="读取短期记忆异常")
def get_short_term_history(self, ...): ...

# 等价的手写展开
def get_short_term_history(self, ...):
    try:
        ...原逻辑...
    except AgentException as e:
        _logger.error(f"读取短期记忆异常 | {type(e).__name__}: {e.message} | detail={e.detail}")
        return []
    except Exception as e:
        _logger.error(f"读取短期记忆异常 | 未知异常: {type(e).__name__}: {str(e)}")
        return []
#   同样 6 行样板 × 几十个方法 = 几百行纯重复；装饰器把重复归零
```

---

## 七、追问清单（提前准备好答案）

**Q1：为什么不直接在 API 层用一个大 try/except 兜住所有异常？**
答：粒度和策略都丢了。全局兜底只能"一视同仁地 500"，没法区分"记忆召回失败应该静默降级"和"记忆写入失败必须让上层知道"；而且异常飞到 API 层时，中间环节的上下文（哪个方法、什么入参）已经丢失，日志只剩一个堆栈。装饰器在**离故障最近的地方**记录，log_msg 直接标明环节。

**Q2：reraise=True 时把 KeyError 包装成 AgentException，原始堆栈会不会丢？**
答：不会。Python 的 `raise X(...)` 在 except 块里执行时会自动设置 `__context__`（隐式异常链），打印堆栈时能看到 "During handling of the above exception..." 两段都保留。若要更显式可以 `raise AgentException(...) from e`。而 AgentException 分支的裸 `raise`（exceptions.py:112）是原样重抛，堆栈纹丝不动。

**Q3：吞异常会不会把 bug 藏起来，让问题更难发现？**
答：会，所以分工明确：**只有"缺了也能跑"的读路径才允许吞**（记忆召回失败 = 这轮没有记忆，对话质量略降但不中断），且吞之前必记 error 级日志，事后可审计；**写路径一律 reraise**，宁可让上层走降级也不让"写入失败"无声消失。这是用代码注释固化的约定（pg_manager.py:651 "异常返回 []，宁可没有上下文也不中断对话"）。

**Q4：装饰器会不会掩盖函数的真实签名，影响 IDE 跳转和文档？**
答：两个手段防这个：① `@functools.wraps(func)`（exceptions.py:100）把原函数的 `__name__`/`__doc__` 复制到 wrapper 上；② `ParamSpec`/`TypeVar` + 两个 `@overload`（exceptions.py:63-80）让类型检查器看到包装后的函数参数与返回值类型与原函数一致，IDE 补全和 mypy 检查都不退化。

**Q5：为什么 default 参数类型标注成 Any 而不是 R（返回值类型）？**
答：exceptions.py:65-67 的注释专门解释了：如果 default 参与 R 的类型推断，写 `@handle_exception(default=None)` 时 R 会被锁成 None，于是"被装饰函数必须返回 None"，所有带返回值的函数全部报类型不兼容。标成 Any 后，R 只从被装饰函数本身推断，包装后类型原样保留。

**Q6：装饰器叠加时顺序有讲究吗？比如和 @retry 叠用。**
答：有。装饰器从下往上包、从上往下执行。`@retry` 在上、`@handle_exception` 在下：每次重试的异常都被 handle_exception 兜底，重试机制永远收不到异常，等于白配。正确顺序是 `@handle_exception` 在上、`@retry` 在下——让重试器先看到原始异常、耗尽次数后再交给 handle_exception 记日志兜底。

---

## 八、简单总结

1. 一个装饰器同时做完三件事：**记日志**（log_msg 定位环节）、**给兜底**（default 与返回类型一致）、**定去留**（reraise 切换吞/抛）
2. 读/写分界线：读方法 `default=[]`/`None`（**宁可没有也不中断对话**），写方法 `reraise=True`（**失败必须让上层降级**，静默吞掉等于丢失失败信号）
3. 最大的坑是兜底值类型不一致：返回 list 的方法兜底必须给 `[]`，给 None 会让调用方的 for 循环当场 TypeError——兜底自己变成新崩溃源
