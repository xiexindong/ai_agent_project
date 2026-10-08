"""
远程沙箱服务（模拟实现：本机就能跑起来的"远端"）
==================================================
给 RemoteSandboxRunner（modules/harness/sandbox.py 的 mode=remote）当对端用的最小服务，
解决"想试远程沙箱但没有第二台机器"的问题。

用法（一般不手工执行，正常由项目命令拉起）：

    python main.py sandbox-server --host 127.0.0.1 --port 8899
    # 等价于
    uvicorn modules.harness.remote_sandbox_server:app --host 127.0.0.1 --port 8899
    python -m modules.harness.remote_sandbox_server

【完整数据流（Agent 主进程 ↔ 本服务 ↔ worker 子进程）】
    1. Agent 主进程：ReAct 判定要调工具 → Harness 按工具声明选到 mode=remote
    2. RemoteSandboxRunner：把工具打包成"类坐标" payload（JSON）
    3. 本服务：POST /run 收到 payload → 校验令牌与 module/class 字段
    4. 本服务：起一个 modules.harness.sandbox_worker 子进程，payload 写进它的 stdin
    5. worker：动态 import 模块 → 取出工具类 → run(params) 真正干活
    6. worker：把结果写成一行 JSON 吐到 stdout（成功失败都是固定结构）
    7. 本服务：取 stdout 里最后一个以 { 开头的合法 JSON 行，原样回给调用方
    8. Agent 主进程：拿到 {"success","result","error","duration_ms"} → ToolResult → 回喂 LLM
    9. 任一步连不上 / 超时 / 非 2xx → 客户端按 fallback 决定降级本机子进程还是直接失败

【四个设计目的】
    1. 位置隔离：工具不在 Agent 主服务里跑，也不在本机子进程里跑，而在"另一个服务"上跑，
       换真远程时把本服务部署到另一台机器即可，协议一行都不用改
    2. 同构协议：与子进程沙箱传的是同一份 {"module","class","params"}，
       远端只要有同一批工具类就能执行，新增工具不用改本文件
    3. 复用执行器：真正的执行仍交给 sandbox_worker，本服务只做"HTTP ↔ 子进程"的转接，
       不重复实现 import→run→封装这套逻辑
    4. 可降级：本服务不可用 ≠ 工具不能跑，客户端会退回本机子进程（沙箱是加固项，不是依赖）

【为什么工具执行失败也返回 HTTP 200】
    200 表示"服务受理了并按协议回了包"，工具跑挂是业务失败，放在 result 的 success/error 里；
    只有"服务本身不可用 / 鉴权不过"才用非 2xx ——
    客户端正是靠这个区分"要不要降级"（工具报错不降级，服务故障才降级）。

【输入输出对照】
    POST /run  in   {"module":"modules.tools.calculator","class":"CalculatorTool",
                     "params":{"expression":"3*7"}}
              out   {"success": true, "result": 21, "error": "", "duration_ms": 1}
    POST /run  in   {"module":"不存在","class":"X","params":{}}
              out   {"success": false, "result": null, "error": "ModuleNotFoundError: ...", ...}
    GET  /health →  {"status":"ok","mode":"remote-mock","worker":"modules.harness.sandbox_worker"}
"""
from __future__ import annotations           # 允许类型注解里用未导入的名字（Python 3.7+ 兼容写法）

import json                                  # 标准库 JSON：payload 打包与结果解析
import os                                    # 标准库 OS：读环境变量（令牌/超时/端口）
import subprocess                            # 标准库子进程：起 sandbox_worker
from pathlib import Path                     # 路径对象：定位项目根目录
from typing import Any, Dict, Optional       # 标准库类型：Any 任意 / Dict 字典 / Optional 可空

from fastapi import FastAPI, Header, HTTPException  # FastAPI 框架：应用 / 请求头取参 / HTTP 异常
from pydantic import BaseModel, Field        # pydantic：请求体模型与字段定义

from modules.harness.sandbox import SubprocessRunner  # 复用子进程执行器（解释器路径 + 环境白名单）

# 【什么】项目根目录（绝对路径）
# 【作用】worker 子进程必须在项目根下起，否则 import 不到 modules.tools.* 这些工具类；
#   本文件位于 <root>/modules/harness/ 下，所以上两级就是项目根
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])  # 本文件上两级 = 项目根，转字符串备用


# 【什么】/run 的请求体
# 【作用】与子进程沙箱的 payload 完全同构：module + class + params
#   【为什么字段名叫 class_name】class 是 Python 关键字，不能直接当属性名，
#   所以属性用 class_name，对外 JSON 仍叫 "class"（靠 alias="class" 对上）
# 【为什么】请求体建模后 FastAPI 自动做类型校验与文档生成，手写解析全省了
class RunRequest(BaseModel):
    module: str = Field(default="", description="工具所在模块，如 modules.tools.calculator")  # 模块路径
    class_name: str = Field(default="", alias="class", description="工具类名，如 CalculatorTool")  # 类名（alias 对上 JSON 的 "class"）
    params: Dict[str, Any] = Field(default_factory=dict, description="工具入参")  # 实参，默认空 dict


app = FastAPI(title="远程沙箱服务（模拟）", version="1.0")  # FastAPI 应用实例；uvicorn 加载的就是它


# 【什么】健康检查
# 【作用】RemoteSandboxRunner.available() 探活用（装配时先探一次，连不上就早降级）；
#         也是"服务到底起没起来"最快的确认方式：curl /health 有回包就是活的
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mode": "remote-mock", "worker": "modules.harness.sandbox_worker"}  # 固定回包：状态/模式/worker 入口


# 【什么】执行一个工具（远端真正的入口）
# 【作用】收类坐标 → 起 worker 子进程 → 回结果 JSON；工具业务性失败也回 200
# 【为什么】200 = 服务受理；工具业务失败放 success/error 字段，客户端据此区分"要不要降级"
@app.post("/run")
def run(req: RunRequest,
        x_sandbox_token: Optional[str] = Header(default=None, alias="X-Sandbox-Token")) -> dict:
    # 【什么是 Header(default=None)】令牌不是必填：FastAPI 会去 HTTP 头里取 X-Sandbox-Token，
    #   客户端没带就是 None，交给 _check_token 决定要不要拒绝
    _check_token(x_sandbox_token)                          # 先鉴权：令牌不对直接 401

    # 【为什么这里只拦 400 的字段缺失】module/class 缺了是调用方用错协议，属于请求不合法，
    #   让它早失败比让 worker 报 ModuleNotFoundError 更好排查
    if not req.module or not req.class_name:               # 缺模块路径或类名
        raise HTTPException(status_code=400, detail="缺少 module / class 字段")  # 400：协议用错

    return execute_payload({                               # 组装 payload 并交给执行出口
        "module": req.module,                              # 模块路径
        "class": req.class_name,                           # 类名（转回对外协议的 "class" 键）
        "params": dict(req.params or {}),                  # 实参副本；None 兜底空 dict
    })


# 【什么】可选的令牌校验
# 【作用】没设 REMOTE_SANDBOX_TOKEN 就不鉴权（本地调试省事）；设了就必须带 X-Sandbox-Token
# 【为什么】令牌来自环境变量而非配置文件：密钥不落盘，换环境不用改代码
def _check_token(token: Optional[str]) -> None:
    expected = str(os.getenv("REMOTE_SANDBOX_TOKEN", "") or "").strip()  # 读期望令牌；空串 = 不启用鉴权
    if expected and token != expected:                     # 启用了鉴权且令牌不匹配
        raise HTTPException(status_code=401, detail="沙箱令牌无效")  # 401：鉴权失败


# 【什么】单次执行的超时（秒）
# 【作用】服务启动前可用 REMOTE_SANDBOX_TIMEOUT_SECONDS 调；
#   max(1.0, ...) 是因为 0 或负数会让 subprocess.run 立刻超时——那不是"不限时"，是"必挂"
# 【为什么】每次调用现读环境变量：改超时不用重启服务进程（调试期友好）
def _timeout_seconds() -> float:
    return max(1.0, float(os.getenv("REMOTE_SANDBOX_TIMEOUT_SECONDS", "30") or 30))  # 环境变量读超时；默认 30s，保底 1s


# 【什么】真正执行 payload，返回可直接 JSON 化的结果字典
# 【作用】HTTP 层唯一的执行出口；结果格式与 sandbox_worker 完全一致，
#         所以调用方（RemoteSandboxRunner）可以直接按子进程沙箱的字段解析
#   【输入输出对照】
#     in  {"module":"modules.tools.calculator","class":"CalculatorTool","params":{"expression":"3*7"}}
#     out {"success": true, "result": 21, "error": "", "duration_ms": 1}
def execute_payload(payload: Dict[str, Any]) -> dict:
    # 【对端是谁】下面这条命令起的就是 modules/harness/sandbox_worker.py：
    #   本服务 → 把 payload 写进 worker 的 stdin → worker 动态 import、run()
    #   → worker 输出一行 JSON → 这里取最后一行合法 JSON 回给调用方
    #   【为什么复用 SubprocessRunner】不是为了跑工具，而是为了复用它两样东西：
    #     runner.python       当前解释器路径（虚拟环境里写死 "python" 会指错解释器）
    #     runner._clean_env() 环境变量白名单清洗（同包内复用，避免把宿主口令带进 worker）
    runner = SubprocessRunner(project_root=PROJECT_ROOT, timeout_seconds=_timeout_seconds())  # 建一个子进程执行器借用其配置
    cmd = [runner.python, "-m", "modules.harness.sandbox_worker"]  # 以模块方式启动 worker

    try:
        # 【六个参数各自的硬性理由 / 不这么写会怎样】
        #   input=...              把 payload 写进 worker 的 stdin；不用临时文件中转
        #   capture_output=True    同时收 stdout/stderr；只收 stdout 的话工具崩了没线索
        #   text=True + encoding   按 UTF-8 解码；不指定 encoding 时 Windows 按 GBK 解，中文结果变乱码
        #   errors="replace"       解码出错时替换而不是抛异常，保证"结果行"永远拿得到
        #   timeout=...            工具死循环时杀掉子进程，不让一个请求拖死整个服务
        #   cwd=项目根             worker 要 import 本项目模块，工作目录不对就 ModuleNotFoundError
        #   env=白名单             只留必要环境变量，别把数据库口令等宿主秘密带进沙箱
        proc = subprocess.run(
            cmd, input=json.dumps(payload, ensure_ascii=False),  # stdin 喂 payload（中文不转义）
            capture_output=True, text=True, encoding="utf-8", errors="replace",  # 收双管道 + utf-8 解码 + 坏字节替换
            timeout=runner.timeout_seconds, cwd=str(runner.project_root),  # 超时即杀 + 项目根工作目录
            env=runner._clean_env(),                           # 白名单环境变量
        )
    except subprocess.TimeoutExpired:                        # worker 超时被杀
        return {"success": False, "result": None,
                "error": f"沙箱执行超时({runner.timeout_seconds:g}s)", "duration_ms": 0}  # 固定结构回包：超时
    except Exception as e:                                   # worker 压根没起起来
        return {"success": False, "result": None,
                "error": f"沙箱启动失败: {type(e).__name__}: {e}", "duration_ms": 0}      # 固定结构回包：启动失败
    return _last_json_line(proc)                             # 从 stdout 捞结果行回给调用方


# 【什么】从 worker 的 stdout 里取最后一行合法 JSON
# 【作用】工具自己 print 的调试信息不会污染结果行（与子进程沙箱同一个约定：以 { 开头才算结果行）
# 【为什么】从后往前扫：worker 契约是"结果写在最后一行"，命中即返回最省事
def _last_json_line(proc) -> dict:
    for line in reversed((proc.stdout or "").splitlines()):  # 从最后一行往前扫；stdout 为 None 兜底空串
        line = line.strip()                                  # 去首尾空白：缩进过的 JSON 也能认
        if not line.startswith("{"):                         # 预筛：不以 { 开头的不是结果行
            continue                                         # 跳过 print/日志行
        try:
            return json.loads(line)                          # 解析成功即返回（命中结果行）
        except json.JSONDecodeError:                         # 以 { 开头但不是合法 JSON
            continue                                         # 继续往前找
    # 【走到这里说明什么】worker 一行 JSON 都没吐出来：被强杀 / import 就炸 / 输出被 stderr 吃了
    #   把 stderr 尾巴带上，调用方才知道是"工具的问题"还是"服务的问题"
    detail = (proc.stderr or proc.stdout or "").strip()[-300:]  # 诊断信息：stderr 优先，末尾 300 字
    return {"success": False, "result": None, "error": f"worker 无有效输出: {detail}", "duration_ms": 0}  # 固定结构回包：无输出


# 【什么】脚本入口
# 【作用】直接 `python -m modules.harness.remote_sandbox_server` 时把服务跑起来；
#   uvicorn 延迟导入（放函数里），不跑服务就不引入这个依赖
# 【为什么】端口走环境变量：与命令行拉起的两种方式（main.py / uvicorn CLI）都能对上台账
if __name__ == "__main__":
    import uvicorn                                       # 延迟导入 ASGI 服务器

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("REMOTE_SANDBOX_PORT", "8899") or 8899))  # 起服务；端口默认 8899
