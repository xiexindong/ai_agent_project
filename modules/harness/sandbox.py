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
from __future__ import annotations           # 允许类型注解里用未导入的名字（Python 3.7+ 兼容写法）

import json                                  # 标准库 JSON：payload 打包与结果解析
import os                                    # 标准库 OS：读宿主环境变量做白名单清洗
import shutil                                # 标准库 shutil：which() 探测 docker 是否存在
import subprocess                            # 标准库子进程：起 worker / 起容器
import sys                                   # 标准库 sys：取当前 Python 解释器路径
import urllib.error                          # 标准库 URL 错误类型：区分 HTTP 错误与连接错误
import urllib.request                        # 标准库 HTTP 客户端：远程沙箱 POST 请求
from abc import ABC, abstractmethod          # 抽象基类工具：定义 SandboxRunner 契约
from pathlib import Path                     # 路径对象：定位项目根目录
from typing import Any, Dict, Optional       # 标准库类型：Any 任意 / Dict 字典 / Optional 可空

from modules.common.logger import get_logger # 统一日志工厂
from modules.tools.base_tool import ToolResult  # 工具执行结果统一结构

_logger = get_logger("harness.sandbox")      # 模块级 logger，名字 = "harness.sandbox"

MODE_NONE = "none"                           # 模式常量：进程内执行（默认，零开销）
MODE_SUBPROCESS = "subprocess"               # 模式常量：本机独立子进程执行
MODE_DOCKER = "docker"                       # 模式常量：docker 容器执行
MODE_REMOTE = "remote"          # 远程沙箱服务（HTTP 协议，见 RemoteSandboxRunner）

# 子进程环境变量白名单：其余一律不继承（避免把数据库口令等带进沙箱）
_ENV_ALLOW = ("PATH", "PYTHONPATH", "LANG", "PYTHONIOENCODING", "TZ", "HOME", "SYSTEMROOT", "TEMP", "TMP")  # 允许带进子进程的环境变量名单


# 【什么】沙箱执行器基类
# 【作用】统一 run(tool, params) → ToolResult，上层不感知在哪执行
# 【为什么】抽象出统一契约后，四档隔离可以按声明互换，Harness 不需要 if/else 分支
class SandboxRunner(ABC):
    """工具执行环境抽象。"""

    mode = MODE_NONE                         # 子类各自覆盖成自己的模式名

    # 【什么】执行入口（抽象方法）
    # 【作用】所有子类必须实现：执行工具并返回统一 ToolResult
    @abstractmethod
    def run(self, tool: Any, params: Dict[str, Any], timeout_seconds: float = 0.0) -> ToolResult:
        """执行工具并返回统一结果。"""


# 【什么】进程内执行（默认）
# 【作用】等价于改造前的行为：直接 tool.run(params)
# 【为什么】只读工具零开销，不为安全性牺牲"查天气"这类调用的毫秒级延迟
class InProcessRunner(SandboxRunner):
    mode = MODE_NONE                         # 模式名：none

    # 【什么】进程内直接执行
    # 【作用】复制一份参数后直接调工具的 run()
    def run(self, tool: Any, params: Dict[str, Any], timeout_seconds: float = 0.0) -> ToolResult:
        return tool.run(dict(params or {}))  # dict() 拷贝防工具改动上游参数；None 兜底成空 dict


# 【什么】子进程执行
# 【作用】崩溃/死循环不影响主进程（超时即杀），环境变量只剩白名单
# 【为什么】有副作用的工具放子进程里跑：它崩了、死循环了，主进程 kill 掉就行
class SubprocessRunner(SandboxRunner):
    """独立子进程执行器。

    【输入输出对照】
        输入  tool=CalculatorTool(), params={"expression": "3*7"}
        过程  起 `python -m modules.harness.sandbox_worker`，stdin 传类坐标 + 参数
        输出  ToolResult(success=True, result=21, duration_ms=...)
        异常  子进程超时被 kill → ToolResult(success=False, error="沙箱执行超时(10s)")
    """

    mode = MODE_SUBPROCESS                   # 模式名：subprocess

    # 【什么】构造函数
    # 【作用】定项目根目录、默认超时、Python 解释器路径
    def __init__(self, project_root: Optional[str] = None, timeout_seconds: float = 15.0,
                 python_executable: Optional[str] = None):
        self.project_root = Path(project_root or self._default_root()).resolve()  # 项目根：没传就按本文件位置推；resolve 转绝对路径
        self.timeout_seconds = max(1.0, float(timeout_seconds))  # 默认超时保底 1s，防传 0 永不超时
        self.python = python_executable or sys.executable or "python"  # 解释器：显式传 > 当前进程的解释器 > PATH 里的 python

    # 【什么】推算项目根目录
    # 【作用】worker 子进程的 cwd 和 PYTHONPATH 都要用它
    @staticmethod
    def _default_root() -> Path:
        # 本文件在 <root>/modules/harness/sandbox.py → 上两级即项目根
        return Path(__file__).resolve().parents[2]     # 上两级：harness → modules → 项目根

    # 【什么】清洗环境变量
    # 【作用】只留白名单，避免把宿主的环境变量（含口令）带进沙箱
    # 【为什么】子进程默认继承全部环境变量，数据库口令/密钥会泄露给工具代码
    def _clean_env(self) -> Dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k.upper() in _ENV_ALLOW}  # 只保留白名单内的环境变量（大小写不敏感）
        env["PYTHONPATH"] = str(self.project_root)         # 强制设 PYTHONPATH：worker 才能 import 到项目模块
        env["PYTHONIOENCODING"] = "utf-8"                  # 强制 utf-8：防 Windows 子进程 gbk 乱码
        return env                                         # 返回清洗后的环境表

    # 【什么】子进程执行入口
    # 【作用】打包类坐标 → 起 worker 子进程 → 解析回传的 JSON 结果
    def run(self, tool: Any, params: Dict[str, Any], timeout_seconds: float = 0.0) -> ToolResult:
        payload = json.dumps({                             # 打包"类坐标"：跨进程只能传字符串，不能传对象
            "module": type(tool).__module__,               # 工具类所在模块，如 "modules.tools.calculator"
            "class": type(tool).__name__,                  # 工具类名，如 "CalculatorTool"
            "params": dict(params or {}),                  # 实参副本；None 兜底成空 dict
        }, ensure_ascii=False)                             # 不转义非 ASCII：中文参数原样传，worker 端好读
        timeout = float(timeout_seconds or self.timeout_seconds)  # 本次超时：调用方指定优先，否则用构造时的默认

        # 【对端是谁】这条命令启动的就是 modules/harness/sandbox_worker.py 的 main()：
        #   本进程 → 把 payload 写进子进程 stdin
        #   → worker 动态 import 模块、取出工具类 → run(params) 真正执行
        #   → worker 把一行 JSON 写回 stdout → 本进程在 _parse() 取最后一行合法 JSON
        # 【为什么传"类坐标"而不是工具对象】工具对象没法 pickle 过进程边界，
        #   传 module + class 名字，让子进程自己按名字加载。
        cmd = [self.python, "-m", "modules.harness.sandbox_worker"]  # 以模块方式启动 worker（等价于 python -m ...）
        try:
            # 【为什么显式 encoding】Windows 下默认用 gbk 解码子进程输出，
            #   工具日志里只要有中文就会抛 UnicodeDecodeError，把一次正常调用搞成失败。
            proc = subprocess.run(
                cmd, input=payload, capture_output=True, text=True,  # 命令行 + stdin 喂 payload + 捕获 stdout/stderr + 文本模式
                encoding="utf-8", errors="replace",                  # 强制 utf-8 解码；坏字节替换而不是抛错
                timeout=timeout, cwd=str(self.project_root), env=self._clean_env(),  # 超时即杀 + 工作目录 + 白名单环境
            )
        except subprocess.TimeoutExpired:                  # 子进程超时被 kill
            _logger.warning(f"沙箱执行超时({timeout}s): {getattr(tool, 'name', '?')}")  # 告警：哪个工具超时
            return ToolResult(success=False, error=f"沙箱执行超时({timeout:g}s)")       # 失败结果：超时
        except Exception as e:                             # 子进程压根没起起来（解释器不存在/权限等）
            return ToolResult(success=False, error=f"沙箱启动失败: {type(e).__name__}: {e}")  # 失败结果：启动失败，带异常类型
        return self._parse(proc, tool)                     # 从子进程 stdout 里解析结果行

    # 【什么】解析子进程输出
    # 【作用】stdout 里可能混着工具自己的 print，取最后一个合法 JSON 行
    # 【为什么】worker 的契约是"结果写在最后一行"，从后往前找第一个合法 JSON 即结果
    @staticmethod
    def _parse(proc, tool: Any) -> ToolResult:
        """从子进程 stdout 里捞出 worker 写的那行结果 JSON。

        【为什么 stdout 会"脏"】worker 与工具跑在同一个子进程里，
            工具内部只要 print 调试信息，文本也会进 stdout：
                Calculating 3*7...
                {"success": true, "result": 21, ...}
            worker 的契约是"结果写在最后一行"，所以这里从后往前找第一个合法 JSON。

        【输入输出对照】
            输入 stdout="Calculating...\n{\"success\": true, \"result\": 21,\n
                         \"error\": \"\", \"duration_ms\": 1}\n"
            输出 ToolResult(success=True, result=21, duration_ms=1)
            输入 stdout="hello world"（没有 JSON 行）
            输出 ToolResult(success=False, error="沙箱无有效输出: hello world")
        """
        # reversed：从最后一行往前扫 —— 结果行约定在末尾，工具的 print 都在它前面
        # proc.stdout or ""：进程异常时 stdout 可能为 None，用 or 兜成空串防 splitlines 报错
        for line in reversed((proc.stdout or "").splitlines()):  # 从后往前逐行扫描
            line = line.strip()                                  # 去掉首尾空白/换行：缩进过的 JSON 也能认
            if not line.startswith("{"):                         # 快速预筛：不是 { 开头的一定不是结果 JSON，直接跳过
                continue                                         #   （print 文本、日志行都在这里被滤掉，省一次 json.loads）
            try:
                data = json.loads(line)                          # 正式解析这一行
            except json.JSONDecodeError:                         # 以 { 开头但不是完整合法 JSON（半截日志/打印了半个字典）
                continue                                         #   继续往前找上一行，不在这里失败
            # 命中合法 JSON → 按 worker 的字段契约还原成 ToolResult
            return ToolResult(
                success=bool(data.get("success")),               # bool()：JSON 里可能是 null/缺字段，统一成确定的 True/False
                result=data.get("result"),                       # 工具产出（数字/字符串，原样保留，不在这里转类型）
                error=str(data.get("error") or ""),               # 失败原因；null/缺失转空串
                duration_ms=int(data.get("duration_ms") or 0),    # 耗时；缺失记 0
            )
        # 整个 stdout 扫完都没有合法 JSON：worker 根本没按契约输出（崩溃在 import 阶段等）
        # detail 优先取 stderr（报错栈通常在那），没有再取 stdout；[-300:] 截断防超长刷屏
        detail = (proc.stderr or proc.stdout or "").strip()[-300:]  # 取诊断信息（stderr 优先），末尾 300 字
        return ToolResult(success=False, error=f"沙箱无有效输出: {detail}")  # 失败结果：无有效输出 + 诊断


# 【什么】容器执行
# 【作用】网络隔离 + 内存限制，出错只影响容器内部；docker 不可用自动降级子进程
# 【为什么】不可信逻辑（执行生成代码等）需要硬隔离；降级保证"加固项不成为可用性瓶颈"
class DockerRunner(SubprocessRunner):
    """容器沙箱执行器。

    【docker 不可用怎么办】
        降级为子进程并在日志里 warning——沙箱是加固项，不该成为可用性瓶颈，
        与项目里"降级而不是失败"的一贯原则一致。
    """

    mode = MODE_DOCKER                       # 模式名：docker

    # 【什么】构造函数
    # 【作用】在子进程执行器基础上追加容器专属参数（镜像/网络/内存/CPU）
    def __init__(self, image: str = "python:3.11-slim", timeout_seconds: float = 30.0,
                 project_root: Optional[str] = None, network: str = "none",
                 memory: str = "256m", cpus: str = "0.5"):
        super().__init__(project_root=project_root, timeout_seconds=timeout_seconds)  # 复用父类：项目根 + 超时 + 解释器
        self.image = image or "python:3.11-slim"           # 容器镜像；空串兜底成 slim 基础镜像
        self.network = network or "none"                   # 容器网络模式；默认 none = 断网
        self.memory = memory or "256m"                     # 内存上限；默认 256MB
        self.cpus = cpus or "0.5"                          # CPU 配额；默认半个核

    # 【什么】docker 是否可用
    # 【作用】装配时据此决定是否降级，避免每次调用都试一次
    # 【为什么】which() 只查 PATH 里有没有 docker 命令，是最便宜的探活方式
    def available(self) -> bool:
        return shutil.which("docker") is not None          # PATH 里找得到 docker 命令 → 可用

    # 【什么】容器执行入口
    # 【作用】docker 可用就走容器，不可用/启动失败/无输出三处降级子进程；只有超时不降级
    def run(self, tool: Any, params: Dict[str, Any], timeout_seconds: float = 0.0) -> ToolResult:
        # ------------------------------------------------------------------
        # 【什么】容器沙箱执行：把工具的"类名 + 参数"打包成 JSON，喂给 docker 容器里
        #        跑的 sandbox_worker，由它 import 类、调用、把结果以 JSON 打回 stdout
        # 【流程】docker 可用？→ 打包 payload → 拼 docker 命令 → 跑容器 → 解析输出
        #        四个环节里三个可降级（不可用/启动失败/无输出），只有超时不重试
        #
        # 【为什么传类名而不是传对象】容器是独立进程，对象没法跨进程传递；
        #   传 "modules.tools.data_query" + "DataQueryTool" 字符串，
        #   容器里的 worker 自己 import 加载（与子进程沙箱同一套路）
        #
        # 【docker 命令逐参对照】
        #   --rm          容器跑完自动删除，不留垃圾
        #   -i            保持标准输入打开：payload 从 stdin 喂进去
        #   --network none 断网：工具就算想发请求也发不出去（默认值）
        #   --memory 256m  内存上限：工具写死循环最多吃 256M 就被杀
        #   --cpus 0.5     CPU 减半：独占不了宿主机算力
        #   -v 项目:/app   把项目目录挂进容器，worker 才 import 得到工具代码
        #   -w /app        容器内工作目录设为挂载点
        #
        # 【降级三处 vs 不降级一处】
        #   降级（退回子进程沙箱）：docker 没装 / 启动异常 / 无有效输出
        #   不降级：超时——重跑大概率还是超时，直接报失败省时间
        #
        # 【输入输出对照】tool=DataQueryTool()、params={"sql":"SELECT 1"}：
        #   payload = '{"module": "modules.tools.data_query", "class": "DataQueryTool",
        #              "params": {"sql": "SELECT 1"}}'
        #   正常：ToolResult(success=True, result=..., duration_ms=...)
        #   断网容器里跑联网工具：工具报错 → success=False（容器活着，错误在容器内）
        # ------------------------------------------------------------------
        if not self.available():                                # 机器上没装 docker：降级子进程沙箱（隔离弱一档但可用）
            _logger.warning("docker 不可用，容器沙箱降级为子进程沙箱")  # 告警：本次走了降级路径
            return super().run(tool, params, timeout_seconds)   # 用父类的子进程方式执行同一份调用
        payload = json.dumps({                                  # 打包"哪个类 + 什么参数"，跨进程只能传字符串
            "module": type(tool).__module__,                    #   如 "modules.tools.data_query"
            "class": type(tool).__name__,                       #   如 "DataQueryTool"
            "params": dict(params or {}),                       #   实参副本；None 兜底成空 dict
        }, ensure_ascii=False)                                  # 不转义非 ASCII：中文参数原样传
        cmd = [
            "docker", "run", "--rm", "-i",                      # docker run：跑完自动删容器 + 保持 stdin 打开
            "--network", self.network,                          # 默认 none：容器内断网
            "--memory", self.memory,                            # 默认 256m：内存上限
            "--cpus", self.cpus,                                # 默认 0.5：CPU 上限
            "-v", f"{self.project_root}:/app",                  # 项目代码挂进容器，worker 才能 import 工具
            "-w", "/app",                                       # 容器内工作目录设为挂载点
            self.image,                                         # 默认 python:3.11-slim
            "python", "-m", "modules.harness.sandbox_worker",   # 容器里跑的入口：读 stdin 的 JSON，执行工具，stdout 回 JSON
        ]
        timeout = float(timeout_seconds or self.timeout_seconds)  # 本次超时：调用方指定优先，否则用构造时的默认
        try:
            proc = subprocess.run(cmd, input=payload, capture_output=True, text=True,  # 起容器：stdin 喂 payload + 捕获输出 + 文本模式
                                  encoding="utf-8", errors="replace", timeout=timeout)  # utf-8 解码 + 超时即杀容器
        except subprocess.TimeoutExpired:                       # 超时不降级：重试大概率还是超时，直接失败
            return ToolResult(success=False, error=f"容器沙箱执行超时({timeout:g}s)")   # 失败结果：超时
        except Exception as e:                                  # docker 启动失败等异常：降级子进程重试
            _logger.warning(f"容器沙箱失败，降级子进程: {e}")   # 告警：启动失败走降级
            return super().run(tool, params, timeout_seconds)   # 用父类的子进程方式重试同一份调用
        result = self._parse(proc, tool)                        # 从 stdout 里取最后一个合法 JSON 行当结果
        if not result.success and not (proc.stdout or "").strip():  # 失败且 stdout 全空 = 容器根本没跑起来
            _logger.warning("容器沙箱无有效输出，降级子进程重试")  # 告警：容器没跑起来走降级
            return super().run(tool, params, timeout_seconds)   # 用父类的子进程方式重试
        return result                                           # 成功或"失败但有输出"（工具在容器里真跑过并报错）都原样返回


# 【什么】远程沙箱执行
# 【作用】把同一份"类坐标" payload POST 给另一台机器上常驻的执行服务，隔离不在本机
# 【为什么】最高档隔离：工具代码根本不在本机跑；降级机制保证远端不可用时业务不断
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

    mode = MODE_REMOTE                       # 模式名：remote

    # 【什么】构造函数
    # 【作用】定远端地址/令牌/超时，并预建一个本机子进程执行器做降级后备
    def __init__(self, endpoint: str = "", token: str = "", timeout_seconds: float = 15.0,
                 fallback: bool = True, project_root: Optional[str] = None,
                 python_executable: Optional[str] = None):
        self.endpoint = str(endpoint or "").strip()          # 远端 /run 地址
        self.token = str(token or "").strip()                # 可选：X-Sandbox-Token
        self.timeout_seconds = max(1.0, float(timeout_seconds))  # 默认超时保底 1s
        self.fallback = bool(fallback)                       # 连不上时是否降级本机子进程
        # 【对端是谁】降级用的就是本机子进程执行器（SubprocessRunner），
        #   与 remote 传同一份类坐标，只是不走网络
        self._local = SubprocessRunner(project_root=project_root,   # 预建降级后备：本机子进程执行器
                                       timeout_seconds=self.timeout_seconds,  # 与远程用同一档超时
                                       python_executable=python_executable)   # 解释器路径透传

    # 【什么】是否配了远程地址
    # 【作用】没配就直接走本机子进程，连网络都不碰
    # 【为什么】空地址是最常见的"没启用 remote"信号，早判断省一次无效连接
    def configured(self) -> bool:
        return bool(self.endpoint)                           # endpoint 非空才算配置了远程沙箱

    # 【什么】由 /run 地址推出 /health 地址
    # 【作用】探活用；推不出来（自定义路径）就返回空，由真正调用时再判断
    # 【为什么】健康检查是约定俗成的伴生接口，能推就推、推不出不硬猜
    def health_url(self) -> str:
        return self.endpoint[:-4] + "/health" if self.endpoint.endswith("/run") else ""  # 去掉 /run 换成 /health；不是 /run 结尾返回空

    # 【什么】远端服务健康吗
    # 【作用】装配时探一次，连不上就早降级，不用等第一次工具调用才发现
    # 【为什么】把"远端挂了"的发现时机从"第一次业务调用"提前到"装配期"，报错更明确
    def available(self) -> bool:
        if not self.configured():                            # 没配地址
            return False                                     # 直接判不可用
        url = self.health_url()                              # 推出健康检查地址
        if not url:                                          # 推不出来（自定义路径）
            return True                                      # 没约定健康检查地址 → 不探活
        try:
            with urllib.request.urlopen(url, timeout=min(3.0, self.timeout_seconds)) as resp:  # 探活请求，超时取 3s 与默认超时的较小值
                return 200 <= int(getattr(resp, "status", 200) or 200) < 300  # 2xx 才算健康
        except Exception as e:                               # 连接失败/超时/解析异常
            _logger.warning(f"远程沙箱健康检查失败: {url} | {e}")  # 告警：哪个地址、什么错
            return False                                     # 判不可用

    # 【什么】远程执行入口
    # 【作用】打包类坐标 → POST 给远端 → 解析返回；失败按 fallback 决定降级还是报错
    def run(self, tool: Any, params: Dict[str, Any], timeout_seconds: float = 0.0) -> ToolResult:
        if not self.configured():                            # 没配远程地址
            _logger.warning("未配置远程沙箱地址，降级为本机子进程沙箱")  # 告警：走了本机降级
            return self._local.run(tool, params, timeout_seconds)  # 直接用本机子进程执行
        # 【什么】打包成"类坐标"，与子进程沙箱完全同一份协议
        payload = json.dumps({
            "module": type(tool).__module__,                 # 工具类所在模块
            "class": type(tool).__name__,                    # 工具类名
            "params": dict(params or {}),                    # 实参副本；None 兜底成空 dict
        }, ensure_ascii=False)                               # 不转义非 ASCII
        timeout = float(timeout_seconds or self.timeout_seconds)  # 本次超时：调用方指定优先
        try:
            data = self._post(payload, timeout)              # 发 POST 并解析返回的 JSON dict
        except Exception as e:                               # 连接失败 / 超时 / 非 2xx / 非法 JSON
            if not self.fallback:                            # 配置不允许降级
                return ToolResult(success=False, error=f"远程沙箱调用失败: {type(e).__name__}: {e}")  # 失败结果：带异常类型
            _logger.warning(f"远程沙箱调用失败，降级本机子进程: "
                            f"{getattr(tool, 'name', '?')} | {e}")  # 告警：哪个工具、什么错，走降级
            return self._local.run(tool, params, timeout_seconds)  # 降级：本机子进程跑同一份 payload
        return ToolResult(
            success=bool(data.get("success")),               # 成功标志；null/缺失统一成 False
            result=data.get("result"),                       # 工具产出原样保留
            error=str(data.get("error") or ""),              # 失败原因；null/缺失转空串
            duration_ms=int(data.get("duration_ms") or 0),   # 耗时；缺失记 0
        )

    # 【什么】把 payload POST 给远端并解析返回的 JSON
    # 【作用】网络/协议的锅统一往外抛，由 run() 决定降级还是报错
    # 【为什么】网络异常类型多（HTTPError/URLError/超时），集中在一处翻译成 RuntimeError
    def _post(self, payload: str, timeout: float) -> dict:
        # 【为什么用标准库 urllib】沙箱是基础组件，不该再引入 requests/httpx 这类网络依赖
        req = urllib.request.Request(
            self.endpoint, data=payload.encode("utf-8"), method="POST",  # POST 请求体 = utf-8 编码的 payload
            headers={"Content-Type": "application/json; charset=utf-8"},  # 声明 JSON + utf-8
        )
        if self.token:                                       # 配了鉴权令牌
            req.add_header("X-Sandbox-Token", self.token)    # 加鉴权头
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # 发请求（带超时）
                body = resp.read().decode("utf-8", errors="replace")    # 读响应体并按 utf-8 解码（坏字节替换）
        except urllib.error.HTTPError as e:                  # 4xx/5xx：把响应体带上，好看服务端报了什么
            detail = ""                                      # 先兜一个空诊断信息
            try:
                detail = (e.read() or b"").decode("utf-8", errors="replace")[:200]  # 读错误响应体，截断 200 字
            except Exception:                                # 连读错误体都失败
                pass                                         # 忽略，detail 保持空
            raise RuntimeError(f"远程沙箱返回 HTTP {e.code}: {detail}") from e  # 翻成 RuntimeError 往上抛
        except urllib.error.URLError as e:                   # 连不上 / DNS 失败 / 被拒
            raise RuntimeError(f"远程沙箱连接失败: {e.reason}") from e  # 翻成 RuntimeError 往上抛
        return json.loads(body)                              # 解析响应 JSON 返回 dict（非法 JSON 的异常也交给 run() 处理）


# 【什么】按模式构造执行器
# 【作用】配置里写 none/subprocess/docker/remote 四种字符串，这里统一转成对象
# 【为什么】未知模式兜底成 none（不加隔离但可用），避免配置笔误把整个 Harness 搞崩
def build_sandbox(mode: str = MODE_NONE, **kwargs) -> SandboxRunner:
    """按模式名构造沙箱执行器；未知模式按 none 处理（安全兜底=不做额外隔离）。"""
    text = str(mode or MODE_NONE).strip().lower()            # 归一化模式名：去空白、转小写、None 兜底 none
    if text == MODE_SUBPROCESS:                              # 子进程模式
        return SubprocessRunner(**kwargs)                    # 构造子进程执行器（kwargs 透传超时等）
    if text == MODE_DOCKER:                                  # 容器模式
        return DockerRunner(**kwargs)                        # 构造容器执行器（kwargs 透传镜像/网络等）
    if text == MODE_REMOTE:                                  # 远程模式
        return RemoteSandboxRunner(**kwargs)                 # 构造远程执行器（kwargs 透传 endpoint/token 等）
    return InProcessRunner()                                 # none 或未知模式：进程内执行器（无 kwargs）
