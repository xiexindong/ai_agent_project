# `self.harness.capability_of(tool)` 逐词拆解
```python
self.harness.capability_of(tool)
```

## 1. `self.harness`
`self`：当前类实例。
`harness` 是这个类的一个成员属性。
> harness 直译：**运行框架 / 工具调度器 / 执行基座**（Agent工具系统里常用命名）
> 它负责管理所有工具、查询工具安全属性、调度工具执行、沙箱管理。

`self.harness` 就是：**当前实例持有的那个工具调度基座对象**。
它有可能是 `None`，所以前面才有三元表达式做判空。

## 2. `.capability_of`
`capability_of` 是 `harness` 对象上的**一个方法**。
命名直译：**获取【xxx】的能力描述**
方法签名伪代码：
```python
def capability_of(self, tool) -> ToolCapability:
    # 根据传入的 tool 对象，返回这个工具的能力/安全元数据
    ...
```

## 3. `(tool)`
入参：`tool`，就是工具实例，比如 `CalculatorTool()`、`SendNoticeTool()`。

## 合在一起
```python
self.harness.capability_of(tool)
```
👉 **调用 harness 的 capability_of 方法，传入工具对象 tool，返回这个工具的 ToolCapability 能力对象。**

返回结果就是刚才我们看到的：
```python
ToolCapability(name='calculator', risk=low, side_effect=False, sandbox='none')
```

# 完整三元表达式再串一遍
```python
cap = self.harness.capability_of(tool) if self.harness is not None else None
```
逻辑：
1. 如果 `self.harness` **不是 None**：
   - 执行 `self.harness.capability_of(tool)`
   - 拿到 ToolCapability 对象，赋值给 `cap`
2. 如果 `self.harness` **是 None**：
   - 不调用 `.capability_of()`（避免报错 `None.capability_of`）
   - `cap = None`

# 通俗类比
- `harness` = 工具安全管理员
- `capability_of(tool)` = 问管理员：**这个工具是什么权限、风险多大、要不要沙箱、有没有副作用？**
- 返回的 `ToolCapability` = 工具的安全档案卡

# 什么时候 cap 会是 None？
当**没有挂载 harness（管理员不存在）**的时候，就查不到工具能力档案，cap 为 None。代码后面一般会加判断：
```python
if cap is None:
    # 没有能力信息，默认禁止执行 / 走兜底策略
```

# 面试可以直接背的一句话
> `capability_of` 是 harness 基座提供的查询方法，输入工具实例，输出工具的能力与安全元数据 ToolCapability；在调用工具前，框架用这份元数据做风险校验、副作用确认和沙箱选择。

要不要我写这个 `capability_of` 方法的简易模拟实现，帮你看懂内部大概怎么判断并返回 ToolCapability？



# 模拟完整示例（可直接运行）
我们先定义 `ToolCapability` 数据类、Harness类、几个工具类，然后演示 `capability_of`，还有你那一行三元表达式。

```python
from dataclasses import dataclass

# 工具能力描述对象
@dataclass
class ToolCapability:
    name: str
    risk: str
    side_effect: bool | None = None
    sandbox: str | None = None
    data_scopes: tuple | None = None


# 模拟 Harness：工具基座，负责查询工具能力
class Harness:
    def capability_of(self, tool):
        """传入工具实例，返回对应的ToolCapability"""
        if isinstance(tool, CalculatorTool):
            return ToolCapability(
                name="calculator",
                risk="low",
                side_effect=False,
                sandbox="none"
            )
        elif isinstance(tool, DataQueryTool):
            return ToolCapability(
                name="data_query",
                risk="low",
                data_scopes=("users", "orders", "inventory")
            )
        elif isinstance(tool, SendNoticeTool):
            return ToolCapability(
                name="send_notice",
                risk="high",
                side_effect=True,
                sandbox="subprocess"
            )
        else:
            return None


# 定义3个工具类（空壳，只是用来区分类型）
class CalculatorTool:
    pass

class DataQueryTool:
    pass

class SendNoticeTool:
    pass


# 主业务类，里面就有你那一行代码
class AgentRunner:
    def __init__(self, harness=None):
        self.harness = harness

    def get_tool_cap(self, tool):
        # ====== 就是你这一行核心代码 ======
        cap = self.harness.capability_of(tool) if self.harness is not None else None
        return cap


# ---------------- 测试演示 ----------------
if __name__ == "__main__":
    # 1. 有harness的情况
    harness = Harness()
    runner = AgentRunner(harness=harness)

    calc_tool = CalculatorTool()
    cap1 = runner.get_tool_cap(calc_tool)
    print("计算器工具能力：", cap1)

    notice_tool = SendNoticeTool()
    cap2 = runner.get_tool_cap(notice_tool)
    print("通知工具能力：", cap2)

    # 2. harness=None 的情况，cap得到None
    runner2 = AgentRunner(harness=None)
    cap3 = runner2.get_tool_cap(calc_tool)
    print("harness为空时，cap =", cap3)
```

## 运行输出
```
计算器工具能力： ToolCapability(name='calculator', risk='low', side_effect=False, sandbox='none', data_scopes=None)
通知工具能力： ToolCapability(name='send_notice', risk='high', side_effect=True, sandbox='subprocess', data_scopes=None)
harness为空时，cap = None
```

# 重点再看这句
```python
cap = self.harness.capability_of(tool) if self.harness is not None else None
```
- 当 `self.harness` 不为空：调用 `harness.capability_of(tool)` → 返回 ToolCapability 对象
- 当 `self.harness` 是 None：不执行 `.capability_of`，直接 cap = None，防止报错

# 后续业务逻辑示例（拿到cap之后做校验）
```python
def check_and_run_tool(self, tool):
    cap = self.harness.capability_of(tool) if self.harness is not None else None

    if cap is None:
        print("没有工具能力信息，禁止执行")
        return

    # 高风险 / 有副作用，需要人工确认
    if cap.risk == "high" or cap.side_effect is True:
        print(f"⚠️ {cap.name} 高风险/有副作用，需要用户确认后才能执行")
    else:
        print(f"✅ {cap.name} 低风险，可以直接执行")
```

你可以直接复制这段代码到本地 Python 跑一下。
要不要我再加一段：**根据 cap.sandbox 来选择不同的执行方式**的代码？