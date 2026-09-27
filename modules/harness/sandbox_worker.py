"""
沙箱内的工具执行器（子进程 / 容器里跑的那一段）
================================================
用法（由 modules/harness/sandbox.py 调用，一般不手工执行）：

    echo '{"module":"modules.tools.calculator","class":"CalculatorTool","params":{"expression":"3*7"}}' \
        | python -m modules.harness.sandbox_worker
    → {"success": true, "result": "21", "error": "", "duration_ms": 1}

【为什么结果只输出一行 JSON】
    工具内部可能 print 调试信息，主进程取 stdout 里最后一个合法 JSON 行，
    因此这里必须保证结果行是最后一行、且以 { 开头。

【完整数据流（主进程 ↔ 本 worker）】
    1. Agent 主进程：ReAct 判定要调工具 → 打包 payload（JSON）
    2. 主进程起子进程：python -m modules.harness.sandbox_worker，payload 写进 stdin
    3. worker：sys.stdin.read() 拿 payload → 动态 import 模块、取出工具类
    4. worker：实例化工具，调 .run(params) 真正干活（读文件 / 查库 / 调 MCP）
    5. worker：成功或失败都输出一行 JSON 到 stdout
    6. 主进程捕获 stdout，取最后一个以 { 开头的合法 JSON 行
    7. 子进程退出：0=正常，1=异常
    8. 主进程拿到 ToolResult 回喂 LLM，继续 ReAct

【四个设计目的】
    1. 进程隔离：工具死循环 / crash 只干掉子进程，Agent 主服务不崩
    2. 动态加载：不必预先 import 所有工具，主进程传"模块名 + 类名"坐标即可，
       新增工具不用改本文件一行代码
    3. 统一协议：stdin 进 JSON，stdout 出 JSON，成功与失败都是固定结构
    4. 异常兜底：任何异常都捕获，返回标准化错误 JSON，让模型能据此重试或改策略

【输入输出对照】
    stdin  {"module":"modules.tools.file_tool","class":"FileReadTool","params":{"path":"test.txt"}}
    stdout {"success": true, "result": "文件内容...", "error": "", "duration_ms": 1}   退出码 0
    stdin  {"module":"不存在的模块","class":"X","params":{}}
    stdout {"success": false, "result": null, "error": "ModuleNotFoundError: ...", ...} 退出码 1
"""
from __future__ import annotations

import importlib
import json
import sys


def main() -> int:
    """子进程入口。约定：返回 0 = 正常，返回 1 = 出错（即进程退出码）。"""
    try:
        # 【什么】从标准输入读主进程传来的 payload
        # 【作用】主进程 → 管道 → 子进程；拿到的是一段 JSON 字符串
        #   sys.stdin.read()  读全部标准输入，就是主进程打包的那段 JSON
        #   or "{}"           读到空内容时当成空 JSON，避免 json.loads("") 直接报错
        #   json.loads()      JSON 字符串 → Python 字典
        payload = json.loads(sys.stdin.read() or "{}")

        # 【什么】按"模块坐标"动态 import 工具所在的模块
        # 【作用】等价于写 import modules.tools.file_tool，只不过是运行时才导入
        #   为什么必须动态：工具类不固定，主进程传哪个模块名就加载哪个，
        #   写死在文件头就得把所有工具都 import 一遍，新增工具还要改这里
        module = importlib.import_module(payload["module"])          # 按坐标 import 工具类

        # 【什么】从模块里取出工具类本身
        # 【作用】等价于 tool_cls = modules.tools.file_tool.FileReadTool
        #   payload["class"] 是类名字符串，如 "FileReadTool"
        tool_cls = getattr(module, payload["class"])

        # 【什么】实例化工具并真正执行
        # 【作用】这一行就是干活的：读文件 / 调 MCP / 查数据库都在这里发生
        #   tool_cls()                    new 一个工具对象
        #   .run(params)                  把模型生成的入参传进去
        #   payload.get("params") or {}   params 缺失或为 None 时用空字典，防止传 None 进去炸
        #   run() 内部自带参数校验与异常兜底，所以这里不再包一层
        result = tool_cls().run(dict(payload.get("params") or {}))   # run() 内部已做参数校验与异常兜底

        # 【什么】把结果写成一行 JSON 吐回标准输出
        # 【作用】主进程一直在监听 stdout，拿到这行就知道工具跑出了什么
        #   result.to_dict()    工具结果对象 → 字典
        #   json.dumps           字典 → JSON 字符串
        #   ensure_ascii=False   中文原样输出，不被转成 \uXXXX
        #   + "\n"               必须换行，主进程按行解析
        sys.stdout.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
        return 0                                                    # 0 = 正常退出

    except Exception as e:
        # 【什么】兜住上面任何一步的异常
        # 【作用】统一错误出口，保证管道不会因为工具崩溃而断掉
        #   会走到这里的三种情况：
        #     ① payload 格式不对 / 缺 module、class 字段
        #     ② 模块或类不存在（import 失败、getattr 失败）
        #     ③ 工具 run() 内部抛出（即使它自己兜底也可能漏）
        #   为什么必须兜：不兜的话子进程 stderr 一片 traceback、stdout 空，
        #   主进程只能报"沙箱无有效输出"，模型拿不到可用信息去重试
        sys.stdout.write(json.dumps({
            "success": False, "result": None,
            "error": f"{type(e).__name__}: {e}", "duration_ms": 0,
        }, ensure_ascii=False) + "\n")
        return 1                                                    # 1 = 异常退出


# 【什么】脚本入口
# 【作用】直接 `python -m modules.harness.sandbox_worker` 时执行 main()，
#         并把它返回的 0/1 作为进程退出码交给操作系统
if __name__ == "__main__":
    raise SystemExit(main())
