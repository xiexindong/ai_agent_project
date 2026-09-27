"""
工具执行沙箱
============
给工具执行提供隔离环境：工具逻辑出错时，影响范围被锁在沙箱里。

【四档隔离，成本与安全性递增】
    none        进程内执行（默认，与改造前完全一致，零开销）
    subprocess  独立子进程：超时可杀、环境变量只剩白名单、崩了不影响主进程
    docker      容器：网络隔离（--network=none）+ 内存限制，需本机有 docker
    remote      远程沙箱服务：把同一份 payload POST 给另一台机器上常驻的执行服务
                （本项目自带可本地启动的模拟实现 modules/harness/remote_sandbox_server.py，
                 没有第二台机器也能把这条路跑通）

【为什么不默认开容器】
    容器沙箱要拉镜像、起容器，单次调用从毫秒级变成秒级，
    对"查天气/算个数"这类只读工具是纯浪费。
    正确用法是按工具声明：只读工具 none，有副作用的工具 subprocess/docker/remote。

【跨进程怎么执行一个 Python 工具】
    工具是对象，不能直接 pickle 到子进程，所以传的是"类坐标"：
        {"module": "modules.tools.calculator", "class": "CalculatorTool", "params": {...}}
    子进程（modules/harness/sandbox_worker.py）按坐标 import 后 run()，
    结果以一行 JSON 回传。工具若 print 到 stdout，也不会污染结果行。
    远程沙箱传的是同一份坐标，只是把"管道"换成了"HTTP"。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from modules.common.logger import get_logger
from modules.tools.base_tool import ToolResult

_logger = get_logger("harness.sandbox")

MODE_NONE = "none"
MODE_SUBPROCESS = "subprocess"
MODE_DOCKER = "docker"
MODE_REMOTE = "remote"          # 远程沙箱服务（HTTP 协议，见 RemoteSandboxRunner）

# 子进程环境变量白名单：其余一律不继承（避免把数据库口令等带进沙箱）
_ENV_ALLOW = ("PATH", "PYTHONPATH", "LANG", "PYTHONIOENCODING", "TZ", "HOME", "SYSTEMROOT", "TEMP", "TMP")


# 【什么】沙箱执行器基类
# 【作用】统一 run(tool, params) → ToolResult，上层不感知在哪执行
class SandboxRunner(ABC):
    """工具执行环境抽象。"""

    mode = MODE_NONE

    @abstractmethod
    def run(self, tool: Any, params: Dict[str, Any], timeout_seconds: float = 0.0) -> ToolResult:
        """执行工具并返回统一结果。"""


# 【什么】进程内执行（默认）
# 【作用】等价于改造前的行为：直接 tool.run(params)
class InProcessRunner(SandboxRunner):
    mode = MODE_NONE

    def run(self, tool: Any, params: Dict[str, Any], timeout_seconds: float = 0.0) -> ToolResult:
        return tool.run(dict(params or {}))


# 【什么】子进程执行
# 【作用】崩溃/死循环不影响主进程（超时即杀），环境变量只剩白名单
class SubprocessRunner(SandboxRunner):
    """独立子进程执行器。

    【输入输出对照】
        输入  tool=CalculatorTool(), params={"expression": "3*7"}
        过程  起 `python -m modules.harness.sandbox_worker`，stdin 传类坐标 + 参数
        输出  ToolResult(success=True, result=21, duration_ms=...)
        异常  子进程超时被 kill → ToolResult(success=False, error="沙箱执行超时(10s)")
    """

    mode = MODE_SUBPROCESS

    def __init__(self, project_root: Optional[str] = None, timeout_seconds: float = 15.0,
                 python_executable: Optional[str] = None):
        self.project_root = Path(project_root or self._default_root()).resolve()
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.python = python_executable or sys.executable or "python"

    @staticmethod
    def _default_root() -> Path:
        # 本文件在 <root>/modules/harness/sandbox.py → 上两级即项目根
        return Path(__file__).resolve().parents[2]

    # 【什么】清洗环境变量
    # 【作用】只留白名单，避免把宿主的环境变量（含口令）带进沙箱
    def _clean_env(self) -> Dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k.upper() in _ENV_ALLOW}
        env["PYTHONPATH"] = str(self.project_root)
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def run(self, tool: Any, params: Dict[str, Any], timeout_seconds: float = 0.0) -> ToolResult:
        payload = json.dumps({
            "module": type(tool).__module__,
            "class": type(tool).__name__,
            "params": dict(params or {}),
        }, ensure_ascii=False)
        timeout = float(timeout_seconds or self.timeout_seconds)

        # 【对端是谁】这条命令启动的就是 modules/harness/sandbox_worker.py 的 main()：
        #   本进程 → 把 payload 写进子进程 stdin
        #   → worker 动态 import 模块、取出工具类 → run(params) 真正执行
        #   → worker 把一行 JSON 写回 stdout → 本进程在 _parse() 取最后一行合法 JSON
        # 【为什么传"类坐标"而不是工具对象】工具对象没法 pickle 过进程边界，
        #   传 module + class 名字，让子进程自己按名字加载。
        cmd = [self.python, "-m", "modules.harness.sandbox_worker"]
        try:
            # 【为什么显式 encoding】Windows 下默认用 gbk 解码子进程输出，
            #   工具日志里只要有中文就会抛 UnicodeDecodeError，把一次正常调用搞成失败。
            proc = subprocess.run(
                cmd, input=payload, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                timeout=timeout, cwd=str(self.project_root), env=self._clean_env(),
            )
        except subprocess.TimeoutExpired:
            _logger.warning(f"沙箱执行超时({timeout}s): {getattr(tool, 'name', '?')}")
            return ToolResult(success=False, error=f"沙箱执行超时({timeout:g}s)")
        except Exception as e:
            return ToolResult(success=False, error=f"沙箱启动失败: {type(e).__name__}: {e}")
        return self._parse(proc, tool)

    # 【什么】解析子进程输出
    # 【作用】stdout 里可能混着工具自己的 print，取最后一个合法 JSON 行
    @staticmethod
    def _parse(proc, tool: Any) -> ToolResult:
        for line in reversed((proc.stdout or "").splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            return ToolResult(
                success=bool(data.get("success")),
                result=data.get("result"),
                error=str(data.get("error") or ""),
                duration_ms=int(data.get("duration_ms") or 0),
            )
        detail = (proc.stderr or proc.stdout or "").strip()[-300:]
        return ToolResult(success=False, error=f"沙箱无有效输出: {detail}")


# 【什么】容器执行
# 【作用】网络隔离 + 内存限制，出错只影响容器内部；docker 不可用自动降级子进程
class DockerRunner(SubprocessRunner):
    """容器沙箱执行器。

    【docker 不可用怎么办】
        降级为子进程并在日志里 warning——沙箱是加固项，不该成为可用性瓶颈，
        与项目里"降级而不是失败"的一贯原则一致。
    """

    mode = MODE_DOCKER

    def __init__(self, image: str = "python:3.11-slim", timeout_seconds: float = 30.0,
                 project_root: Optional[str] = None, network: str = "none",
                 memory: str = "256m", cpus: str = "0.5"):
        super().__init__(project_root=project_root, timeout_seconds=timeout_seconds)
        self.image = image or "python:3.11-slim"
        self.network = network or "none"
        self.memory = memory or "256m"
        self.cpus = cpus or "0.5"

    # 【什么】docker 是否可用
    # 【作用】装配时据此决定是否降级，避免每次调用都试一次
    def available(self) -> bool:
        return shutil.which("docker") is not None

    def run(self, tool: Any, params: Dict[str, Any], timeout_seconds: float = 0.0) -> ToolResult:
        if not self.available():
            _logger.warning("docker 不可用，容器沙箱降级为子进程沙箱")
            return super().run(tool, params, timeout_seconds)
        payload = json.dumps({
            "module": type(tool).__module__,
            "class": type(tool).__name__,
            "params": dict(params or {}),
        }, ensure_ascii=False)
        cmd = [
            "docker", "run", "--rm", "-i",
            "--network", self.network,
            "--memory", self.memory,
            "--cpus", self.cpus,
            "-v", f"{self.project_root}:/app",
            "-w", "/app",
            self.image,
            "python", "-m", "modules.harness.sandbox_worker",
        ]
        timeout = float(timeout_seconds or self.timeout_seconds)
        try:
            proc = subprocess.run(cmd, input=payload, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error=f"容器沙箱执行超时({timeout:g}s)")
        except Exception as e:
            _logger.warning(f"容器沙箱失败，降级子进程: {e}")
            return super().run(tool, params, timeout_seconds)
        result = self._parse(proc, tool)
        if not result.success and not (proc.stdout or "").strip():
            _logger.warning("容器沙箱无有效输出，降级子进程重试")
            return super().run(tool, params, timeout_seconds)
        return result


# 【什么】远程沙箱执行
# 【作用】把同一份"类坐标" payload POST 给另一台机器上常驻的执行服务，隔离不在本机
class RemoteSandboxRunner(SandboxRunner):
    """远程沙箱执行器：工具在远端跑，本机只发一个 HTTP 请求。

    【和子进程沙箱的区别】
        subprocess  在本机起一个 python 子进程
        remote      POST 给远端执行服务（本项目自带模拟实现
                    modules/harness/remote_sandbox_server.py，本机就能起）
        两者传的东西完全一样 —— {"module","class","params"}，
        所以远端只要能 import 到同一批工具类就能跑，协议不用为工具做任何改动。

    【完整数据流（本执行器 ↔ 远端服务 ↔ worker）】
        1. Harness 按工具声明选到 mode=remote → 调本类的 run()
        2. run()：把工具打包成"类坐标" {"module","class","params"}（与子进程沙箱同一份）
        3. _post()：POST 到远端 /run，带 X-Sandbox-Token（配了才带）
        4. 远端服务（remote_sandbox_server.py）：起 sandbox_worker 子进程执行
        5. worker：动态 import → run(params) → 输出一行 JSON
        6. 远端服务：把那行 JSON 原样回给本执行器
        7. run()：解成 ToolResult 交给 Harness → 回喂 LLM
        8. 任一步连不上 / 超时 / 非 2xx → fallback=True 时降级本机子进程跑同一份 payload

    【输入输出对照】
        输入  endpoint="http://127.0.0.1:8899/run", tool=CalculatorTool(),
              params={"expression": "3*7"}
        过程  POST {"module":"modules.tools.calculator","class":"CalculatorTool","params":{...}}
              ←    {"success": true, "result": 21, "error": "", "duration_ms": 1}
        输出  ToolResult(success=True, result=21, duration_ms=1)
        异常  连不上 / 超时 / 非 2xx → fallback=True 时降级本机子进程，
              否则返回 ToolResult(success=False, error="远程沙箱调用失败: ...")

    【为什么要降级】
        远程是更强的一层隔离，但也是更脆的一层依赖：网络抖一下、服务没起都算失败。
        沙箱是加固项，不该成为可用性瓶颈 —— 与 docker 不可用降级子进程同一个原则。
    """

    mode = MODE_REMOTE

    def __init__(self, endpoint: str = "", token: str = "", timeout_seconds: float = 15.0,
                 fallback: bool = True, project_root: Optional[str] = None,
                 python_executable: Optional[str] = None):
        self.endpoint = str(endpoint or "").strip()          # 远端 /run 地址
        self.token = str(token or "").strip()                # 可选：X-Sandbox-Token
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.fallback = bool(fallback)                       # 连不上时是否降级本机子进程
        # 【对端是谁】降级用的就是本机子进程执行器（SubprocessRunner），
        #   与 remote 传同一份类坐标，只是不走网络
        self._local = SubprocessRunner(project_root=project_root,
                                       timeout_seconds=self.timeout_seconds,
                                       python_executable=python_executable)

    # 【什么】是否配了远程地址
    # 【作用】没配就直接走本机子进程，连网络都不碰
    def configured(self) -> bool:
        return bool(self.endpoint)

    # 【什么】由 /run 地址推出 /health 地址
    # 【作用】探活用；推不出来（自定义路径）就返回空，由真正调用时再判断
    def health_url(self) -> str:
        return self.endpoint[:-4] + "/health" if self.endpoint.endswith("/run") else ""

    # 【什么】远端服务健康吗
    # 【作用】装配时探一次，连不上就早降级，不用等第一次工具调用才发现
    def available(self) -> bool:
        if not self.configured():
            return False
        url = self.health_url()
        if not url:
            return True                                      # 没约定健康检查地址 → 不探活
        try:
            with urllib.request.urlopen(url, timeout=min(3.0, self.timeout_seconds)) as resp:
                return 200 <= int(getattr(resp, "status", 200) or 200) < 300
        except Exception as e:
            _logger.warning(f"远程沙箱健康检查失败: {url} | {e}")
            return False

    def run(self, tool: Any, params: Dict[str, Any], timeout_seconds: float = 0.0) -> ToolResult:
        if not self.configured():
            _logger.warning("未配置远程沙箱地址，降级为本机子进程沙箱")
            return self._local.run(tool, params, timeout_seconds)
        # 【什么】打包成"类坐标"，与子进程沙箱完全同一份协议
        payload = json.dumps({
            "module": type(tool).__module__,
            "class": type(tool).__name__,
            "params": dict(params or {}),
        }, ensure_ascii=False)
        timeout = float(timeout_seconds or self.timeout_seconds)
        try:
            data = self._post(payload, timeout)
        except Exception as e:                               # 连接失败 / 超时 / 非 2xx / 非法 JSON
            if not self.fallback:
                return ToolResult(success=False, error=f"远程沙箱调用失败: {type(e).__name__}: {e}")
            _logger.warning(f"远程沙箱调用失败，降级本机子进程: "
                            f"{getattr(tool, 'name', '?')} | {e}")
            return self._local.run(tool, params, timeout_seconds)
        return ToolResult(
            success=bool(data.get("success")),
            result=data.get("result"),
            error=str(data.get("error") or ""),
            duration_ms=int(data.get("duration_ms") or 0),
        )

    # 【什么】把 payload POST 给远端并解析返回的 JSON
    # 【作用】网络/协议的锅统一往外抛，由 run() 决定降级还是报错
    def _post(self, payload: str, timeout: float) -> dict:
        # 【为什么用标准库 urllib】沙箱是基础组件，不该再引入 requests/httpx 这类网络依赖
        req = urllib.request.Request(
            self.endpoint, data=payload.encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        if self.token:
            req.add_header("X-Sandbox-Token", self.token)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:                  # 4xx/5xx：把响应体带上，好看服务端报了什么
            detail = ""
            try:
                detail = (e.read() or b"").decode("utf-8", errors="replace")[:200]
            except Exception:
                pass
            raise RuntimeError(f"远程沙箱返回 HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:                   # 连不上 / DNS 失败 / 被拒
            raise RuntimeError(f"远程沙箱连接失败: {e.reason}") from e
        return json.loads(body)


# 【什么】按模式构造执行器
# 【作用】配置里写 none/subprocess/docker/remote 四种字符串，这里统一转成对象
def build_sandbox(mode: str = MODE_NONE, **kwargs) -> SandboxRunner:
    """按模式名构造沙箱执行器；未知模式按 none 处理（安全兜底=不做额外隔离）。"""
    text = str(mode or MODE_NONE).strip().lower()
    if text == MODE_SUBPROCESS:
        return SubprocessRunner(**kwargs)
    if text == MODE_DOCKER:
        return DockerRunner(**kwargs)
    if text == MODE_REMOTE:
        return RemoteSandboxRunner(**kwargs)
    return InProcessRunner()
