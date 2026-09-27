# `verdict = self.harness.before_call(req)`
## 一句话概括
**在真正调用工具之前，把本次工具调用请求 `req` 交给 harness 基座做前置校验/安全审核，返回一个审核结果 verdict（裁决）。**

> 类比：`capability_of` 是**查工具本身的静态属性档案**；`before_call(req)` 是**针对这一次具体的调用请求做动态审核**。

---

## 逐段拆解
1. `self.harness`：还是那个工具调度/安全基座，和前面同一个对象
2. `.before_call(req)`：harness 的方法
   - `req`：tool call request，**本次工具调用请求对象**
     里面一般包含：工具名称、入参、调用者ID、上下文、要执行的参数（比如SQL语句、要发送的通知内容）
   - `before_call`：顾名思义，**调用工具之前执行**（before tool call）
3. `verdict`：裁决结果，是审核返回的对象，一般包含：
   - 是否允许执行：`allow: bool`
   - 拒绝理由：`reason: str`
   - 改写后的参数（可选，harness可以拦截并修改请求）

### 伪代码：Verdict 结构
```python
@dataclass
class Verdict:
    allow: bool
    reason: str = ""
    modified_req: ToolCallRequest | None = None
```

## 工作流程（串联前面代码）
```python
# 1. 获取工具静态能力档案
cap = self.harness.capability_of(tool) if self.harness is not None else None

# 2. 发起本次调用的前置安全审核
verdict = self.harness.before_call(req)

# 3. 根据裁决判断能不能跑工具
if verdict.allow:
    # 放行，执行工具
    result = tool.run(req.arguments)
else:
    # 拦截，拒绝执行，返回拒绝原因
    return {"error": verdict.reason}
```

## 和 `capability_of` 的区别（面试重点）
- `capability_of(tool)`：**静态，工具本身的属性**
  工具是什么类型、风险等级、有没有副作用、允许的数据范围。**不看这次调用的参数**。
  > 例：send_notice 工具本身就是 high risk、side_effect=True。

- `before_call(req)`：**动态，针对这一次调用请求**
  拿到本次调用参数做检查：参数是否越权、内容是否违规、是否符合权限策略。
  > 例：send_notice 这次要发给管理员，禁止，直接拦截。

## 给你加到刚才的示例代码里，完整可运行版本
```python
from dataclasses import dataclass

# 工具能力档案
@dataclass
class ToolCapability:
    name: str
    risk: str
    side_effect: bool | None = None
    sandbox: str | None = None
    data_scopes: tuple | None = None

# 工具调用请求
@dataclass
class ToolCallRequest:
    tool_name: str
    arguments: dict

# 前置审核返回的裁决 verdict
@dataclass
class Verdict:
    allow: bool
    reason: str = ""


# Harness 基座，新增 before_call
class Harness:
    def capability_of(self, tool):
        if isinstance(tool, CalculatorTool):
            return ToolCapability(name="calculator", risk="low", side_effect=False, sandbox="none")
        elif isinstance(tool, DataQueryTool):
            return ToolCapability(name="data_query", risk="low", data_scopes=("users", "orders", "inventory"))
        elif isinstance(tool, SendNoticeTool):
            return ToolCapability(name="send_notice", risk="high", side_effect=True, sandbox="subprocess")
        return None

    def before_call(self, req: ToolCallRequest) -> Verdict:
        """工具调用前，对本次请求做动态审核"""
        # 示例规则：如果是发送通知，且消息内容包含"删除"，直接拦截
        if req.tool_name == "send_notice":
            msg = req.arguments.get("message", "")
            if "删除" in msg:
                return Verdict(allow=False, reason="通知内容包含敏感词，禁止发送")
        # 其余情况放行
        return Verdict(allow=True, reason="校验通过")


class CalculatorTool:
    pass
class DataQueryTool:
    pass
class SendNoticeTool:
    pass


class AgentRunner:
    def __init__(self, harness=None):
        self.harness = harness

    def run_tool(self, tool, req: ToolCallRequest):
        # 1. 获取工具静态能力
        cap = self.harness.capability_of(tool) if self.harness is not None else None
        if cap is None:
            return {"error": "无工具能力信息，禁止执行"}

        # 2. 【重点】调用前置安全校验 before_call
        verdict = self.harness.before_call(req)
        print(f"审核结果 verdict: {verdict}")

        if not verdict.allow:
            return {"error": verdict.reason}

        # 审核通过，执行工具
        return {"result": "工具执行成功"}


if __name__ == "__main__":
    harness = Harness()
    runner = AgentRunner(harness=harness)

    # 正常发送通知
    req1 = ToolCallRequest(tool_name="send_notice", arguments={"message": "任务已完成"})
    print(runner.run_tool(SendNoticeTool(), req1))

    # 带敏感词，会被 before_call 拦截
    req2 = ToolCallRequest(tool_name="send_notice", arguments={"message": "删除所有订单"})
    print(runner.run_tool(SendNoticeTool(), req2))
```

运行输出：
```
审核结果 verdict: Verdict(allow=True, reason='校验通过')
{'result': '工具执行成功'}
审核结果 verdict: Verdict(allow=False, reason='通知内容包含敏感词，禁止发送')
{'error': '通知内容包含敏感词，禁止发送'}
```

## 常见 before_call 里面做的校验逻辑
1. 参数合法性校验
2. 数据范围校验（data_scopes，检查SQL是不是查了不在允许范围内的表）
3. 内容安全过滤（敏感词）
4. 权限校验：当前用户能不能调用这个工具
5. 频率限流：调用次数超限直接拒绝

> 后续一般还会配套 `after_call`，工具执行完之后的回调，用来记录日志、审计、检查返回结果。

要不要我把 `after_call` 也一并补上，形成完整工具调用生命周期：`capability_of` → `before_call` → run tool → `after_call`？



# 核心一句话
`before_call(req)` **不是魔法**：
`req`（ToolCallRequest）里面装了**本次调用的全部信息**（工具名、传入参数、上下文），`harness.before_call` 读取 `req` 里面的数据，然后写好的 if 判断规则，得出 `verdict`（允许/拒绝）。

## 先看 req 里面有什么（决定它能判断什么）
```python
@dataclass
class ToolCallRequest:
    tool_name: str          # 是哪个工具：send_notice / data_query
    arguments: dict         # 传给工具的参数！最关键
    caller_id: str = None   # 是谁在调用（用户/agent）
```
例子：
```python
req = ToolCallRequest(
    tool_name="send_notice",
    arguments={"message": "删除全部数据", "receiver": "all_user"}
)
```
`req` 里面已经把**这次要干什么**全部打包好了。`before_call` 的输入就是这个对象。

## 看 before_call 内部是怎么写判断的（拆开看源码）
```python
def before_call(self, req: ToolCallRequest) -> Verdict:
    # 1. 先看是哪个工具，从 req.tool_name 获取
    if req.tool_name == "send_notice":
        # 2. 取出本次调用传入的参数 message
        msg = req.arguments.get("message", "")
        # 3. 写死规则：如果消息包含敏感词 → 拒绝
        if "删除" in msg or "销毁" in msg:
            return Verdict(allow=False, reason="消息包含敏感操作词，禁止发送")
    
    # 数据查询工具的判断例子
    elif req.tool_name == "data_query":
        sql = req.arguments.get("sql", "")
        # 规则：不允许查询 secret 表
        if "secret" in sql:
            return Verdict(allow=False, reason="禁止访问secret表")

    # 所有规则都没命中 → 放行
    return Verdict(allow=True, reason="校验通过")
```

### 判断的逻辑链条
1. 拿到 `req`，读取 `tool_name` → 知道现在要执行哪个工具
2. 读取 `req.arguments` → 拿到这次调用的参数（消息内容 / SQL语句）
3. 用预先写好的**业务安全规则**去匹配参数
4. 匹配上禁止规则 → 返回 `allow=False`；没命中禁止规则 → `allow=True`

> ⚠️ 区分：
> - `capability_of(tool)`：看**工具本身静态属性**（这个工具天生是高风险）
> - `before_call(req)`：看**这一次调用的参数**（这次调用具体想干什么）

## 举两个对比案例
案例1：send_notice，内容正常
```python
req = ToolCallRequest(tool_name="send_notice", arguments={"message": "任务完成"})
# before_call读取message，没有敏感词 → allow=True
```

案例2：send_notice，内容敏感
```python
req = ToolCallRequest(tool_name="send_notice", arguments={"message": "删除全部订单"})
# before_call读取message，发现有“删除” → allow=False，拦截
```

## 扩展：真实框架里 before_call 还能做哪些判断
1. **权限判断**：`req.caller_id` 这个用户有没有权限调用这个工具
2. **数据范围校验**：data_query 的SQL，检查访问的表是否在 `cap.data_scopes` 允许范围内（结合capability_of的结果一起判断）
3. **限流**：同一个用户1分钟最多调用多少次，超出拦截
4. **参数格式校验**：参数缺失、类型不对直接拒绝
5. **内容安全**：调用大模型做内容审核，判断消息是否违规

## 面试一句话总结
> `before_call` 的输入是本次工具调用请求对象`req`，里面包含工具名称和本次调用的全部入参；harness读取req里面的信息，执行预先定义好的安全规则，从而判断本次调用是否允许执行，返回裁决verdict。

## 额外问题：那 before_call 能不能拿到 cap（ToolCapability）？
当然可以！很多框架会在 `before_call` 内部调用 `capability_of`，**组合静态能力+本次请求参数一起判断**。
示例：
```python
def before_call(self, req: ToolCallRequest, tool):
    cap = self.capability_of(tool)
    # 组合判断：工具本身高风险，并且本次参数会修改数据 → 需要人工审批
    if cap.risk == "high" and cap.side_effect:
        return Verdict(allow=False, reason="高风险副作用操作，需要人工确认")
```

要不要我写一个**组合版本**：`capability_of` + `before_call` 联动校验，静态能力+动态参数一起判断？


# 一句话结论
**基础框架骨架是别人写好的（Harness、capability_of、before_call 这套机制）；但是安全规则、判断逻辑，是你来写的。**

我们分开两层理解：
## 1）框架自带的：固定的“机制/流程”
框架已经帮你把整个调用链路搭好了，这套流程是写死在框架里，不用你改：
```
Agent准备调用工具 →
1. 调用 harness.capability_of(tool) 拿到工具静态能力
2. 调用 harness.before_call(req) 做前置校验
3. 判断 verdict 是否允许执行
4. 允许就运行工具；不允许直接拦截
```
框架定义好了：
- `Harness` 这个抽象基类（接口）
- `before_call` 方法**必须接收 req，返回 Verdict 对象**这个规范
- 什么时候自动调用 before_call（工具执行前自动触发）

> 框架只规定：**在这个时间点，会调用这个函数，你返回允许/拒绝**。
> 框架**不会内置你的业务安全规则**。

## 2）由你来实现：Harness 的具体逻辑（规则你写）
框架一般提供一个**抽象基类（Abstract Base Class）**，你继承它，自己重写 `capability_of` 和 `before_call`，在这里写你的业务判断规则。

伪代码（框架自带的抽象类）
```python
# ========== 框架自带（你不用写）==========
from abc import ABC, abstractmethod

class BaseHarness(ABC):
    @abstractmethod
    def capability_of(self, tool):
        """由子类实现：返回工具静态能力"""
        pass

    @abstractmethod
    def before_call(self, req) -> Verdict:
        """由子类实现：你在这里写动态校验规则"""
        pass
```

然后**你自己写子类，实现里面的规则**：
```python
# ========== 你写的代码 ==========
class MyCustomHarness(BaseHarness):
    def capability_of(self, tool):
        # ✅ 你自己定义：每个工具的静态能力
        if isinstance(tool, SendNoticeTool):
            return ToolCapability(name="send_notice", risk="high", side_effect=True, sandbox="subprocess")
        ...

    def before_call(self, req) -> Verdict:
        # ✅【重点】这里全部是你写的动态检查规则！
        # 规则1：send_notice 消息不能包含“删除”
        if req.tool_name == "send_notice":
            msg = req.arguments.get("message", "")
            if "删除" in msg:
                return Verdict(allow=False, reason="禁止发送删除指令")
        
        # 规则2：data_query 不允许查询secret表
        if req.tool_name == "data_query":
            sql = req.arguments["sql"]
            if "secret" in sql.lower():
                return Verdict(allow=False, reason="禁止访问secret表")

        # 全部规则通过
        return Verdict(allow=True)
```

最后把你写好的 `MyCustomHarness` 实例传给 AgentRunner：
```python
harness = MyCustomHarness()  # 用你自己实现的规则
runner = AgentRunner(harness=harness)
```

## 两种模式区分（面试常问）
1. **框架提供机制，用户注入规则（就是咱们现在这套）**
    - 框架：管调用时机、参数封装、Verdict 结构体
    - 你：写业务安全规则，决定放行还是拦截
    ✅ 绝大多数 Agent 框架（LangGraph、自研Agent基座）都是这种模式

2. **框架内置固定规则（很少见）**
    before_call 里面逻辑写死，你不能改。灵活性很差，一般不会这么设计。

## 两个函数分工再总结
- `capability_of`：**静态规则（你定义）**，工具本身的属性，一次性定义好。
- `before_call`：**动态规则（你定义）**，每次工具调用都会跑一遍，读取本次请求参数做检查。

## 举个生活化类比
框架 = 商场安保系统：规定「顾客进门之前必须安检」这个流程。
`before_call` = 安检窗口。
**框架只规定：进门要过安检。但是安检查哪些违禁品，是你定的。**
你可以规定：不能带刀；也可以改成：不能带打火机。安检的判断逻辑由你来写。

## 可选扩展玩法（也是你控制）
你甚至可以在 `before_call` 里面：
- 调用外部权限服务，判断当前用户是否有权限
- 调用 LLM 审核本次工具参数是不是高危意图
- 限流统计，判断这个用户调用次数超限
- 修改请求参数：`modified_req`，把参数清洗之后放行

这些全部都是**由你来实现**，框架只是预留了这个扩展点。

要不要我给你写一个极简的抽象Harness + 自定义实现的完整可运行代码，直观看到「框架基类」和「你的业务规则」分开？