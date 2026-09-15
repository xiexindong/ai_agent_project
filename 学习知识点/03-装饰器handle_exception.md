# 知识点 03：`@handle_exception` 装饰器

> 对应代码：`pg_manager.py:399` + `modules/common/exceptions.py:44`

## 一、被问到的这两行

```399:400:modules/memory/pg_manager.py
    @handle_exception(default=None, log_msg="更新会话状态异常", reraise=True)
    def update_session_state(self, session_id: str, agent_state: dict) -> None:
```

| 行 | 是什么 |
|---|---|
| 399 | **装饰器**：给下面的函数套一层"异常兜底壳" |
| 400 | **函数签名**：声明这个函数要什么、返回什么 |

## 二、第 400 行：函数签名（先说简单的）

```python
def update_session_state(self, session_id: str, agent_state: dict) -> None:
#                        │     │              │                 └─ 返回 None
#                        │     │              └─ 第2个参数：Agent 状态字典
#                        │     └─ 第1个参数：会话ID，字符串
#                        └─ 实例自己（类的方法必须有，调用时不用传）
```

调用时只传后两个：

```python
pg.update_session_state("abc123", {"iteration": 2})
#  ↑ self 由 Python 自动填，你不用管
```

**`-> None` 为什么是 None？** 因为它只写不读——把状态存进去就完事，没有东西要返回给你。对比 `get_session_state(...) -> dict`，那个要读，所以返回字典。

## 三、第 399 行：装饰器在干嘛

### 大白话：给函数雇了个保镖

```
没有装饰器：  调用 → 函数执行 → 崩了 → 异常直接飞到最顶层 → 整个请求 500
有装饰器：    调用 → 保镖接住 → 函数执行 → 崩了被保镖拦下 → 记日志 → 按规矩处理
```

### 它带括号，说明是"能传参的装饰器"

```python
@handle_exception(...)        # ← 带参数，说明它"造"出一个装饰器
def foo(): ...

# 等价于：
def foo(): ...
foo = handle_exception(...)(foo)   # 先调用得到装饰器，再装饰函数
```

所以 `exceptions.py` 里它是**三层嵌套**：

```44:57:modules/common/exceptions.py
def handle_exception(
    default=None,
    log_msg: str = "操作异常",
    reraise: bool = False,
):
    def decorator(func: Callable):      # 第2层：接收被装饰的函数
        @functools.wraps(func)
        def wrapper(*args, **kwargs):   # 第3层：真正被调用的替身
```

| 层 | 作用 | 什么时候执行 |
|---|---|---|
| 第 1 层 `handle_exception(...)` | 收配置参数 | 代码加载时（`import` 那一刻） |
| 第 2 层 `decorator(func)` | 收函数 | 代码加载时 |
| 第 3 层 `wrapper(*args)` | 真正干活 | **每次调用函数时** |

第 3 层用 `*args, **kwargs` 是为了通吃——不管被装饰的函数有几个参数、叫什么名，都能透传。

### 三个参数各管什么

| 参数 | 含义 |
|---|---|
| `default` | 出错时**返回**的兜底值 |
| `log_msg` | 日志前缀，方便 grep 定位 |
| `reraise` | 出错时**是否重新抛出** |

### 内部两条分支

```62:73:modules/common/exceptions.py
            except AgentException as e:
                _logger.error(
                    f"{log_msg} | {type(e).__name__}: {e.message} | detail={e.detail}"
                )
                if reraise:
                    raise
                return default
            except Exception as e:
                _logger.error(f"{log_msg} | 未知异常: {type(e).__name__}: {str(e)}")
                if reraise:
                    raise AgentException(log_msg, str(e))
                return default
```

| 捕获到 | 说明 | 日志 |
|---|---|---|
| `AgentException` | 项目**自己定义**的异常，带 `message` / `detail` | 打全，信息详细 |
| `Exception` | 其他一切未知异常 | 打类型和字符串 |

`AgentException` 是基类，`ToolException` / `MemoryException` / `MCPException` 都继承它（`exceptions.py:31-40`），所以都会被第一条接住。

⚠️ 未知异常在 `reraise=True` 时被**包装成 `AgentException`** 再抛——这样上层只需要 `except AgentException` 就能统一处理，不用去猜 psycopg2、requests、json 各种库的异常类型。

## 四、⚠️ 一个值得注意的细节：`default=None` 在这里是死代码

```python
@handle_exception(default=None, ..., reraise=True)
#                 ↑ 永远不会生效        ↑ 因为这个
```

看 wrapper 的逻辑：两条 `except` 分支都是 `if reraise: raise`，然后 `return default`。**`reraise=True` 时一定先 raise 走了，永远轮不到 `return default`。**

所以这个 `default=None` 写了等于没写（不影响功能，只是容易让人误以为"出错会返回 None"）。

### 对比同文件里的另一个函数

```432:433:modules/memory/pg_manager.py
    @handle_exception(default={}, log_msg="读取会话状态异常")
    def get_session_state(self, session_id: str, user_id: str) -> dict:
```

**没写 `reraise`** → 用默认值 `False` → 出错时**返回 `{}`**，不抛出。

### 那什么时候该 reraise，什么时候该 default？

| 场景 | 配置 | 理由 |
|---|---|---|
| **关键写路径**（`save_turn` / `create_session` / `update_session_state`） | `reraise=True` | 存不进去就是真故障，**不能悄悄吞掉**，必须让上层知道并给用户报错 |
| **可降级的读路径**（`get_session_state` / 向量召回） | 默认 `False` + `default={}` | 读失败不致命，返回空让主流程继续（比如记忆召回失败，就当没历史接着答） |

> 这就是所谓的 **"关键路径 fail-fast，旁路 fail-safe"**。

## 五、`@functools.wraps(func)` 是干嘛的

装饰器的副作用：函数被 `wrapper` 替换后，它的 `__name__` 会变成 `"wrapper"`，`__doc__` 也没了。

后果很实际——本项目 docstring 写得极长（你前面问的那段就在 docstring 里），没有 `wraps` 的话：

```python
help(pg.update_session_state)   # 显示 wrapper 的信息，看不到你写的那份说明
pg.update_session_state.__name__  # 'wrapper'，日志里打出来一脸懵
```

`wraps` 就是把原函数的 `__name__` / `__doc__` / `__module__` 等元信息**复制**到 wrapper 上，让替身看起来跟本尊一样。

## 六、速查表

```python
@handle_exception(...)         # 带参装饰器 = 三层函数，加载时执行外两层
    default   → 出错返回值（仅 reraise=False 时有效）
    log_msg   → 日志前缀
    reraise   → True 抛异常（关键写路径）/ False 返回兜底（可降级读路径）
functools.wraps                # 保留原函数名和 docstring，装饰器的标配
-> None                        # 只写不读的函数就返回 None
```

## 七、一句话带走

> 装饰器 = 在不改函数一行代码的前提下，给它套一层统一处理。
> **`reraise=True` 的写操作出错必须炸出来让人知道；不 reraise 的读操作出错就悄悄降级。**
