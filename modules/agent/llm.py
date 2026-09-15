"""
LLM 客户端封装模块
==================
DeepSeek Chat 对话模型封装：原生 Function Calling、结构化输出、重试容错。
覆盖边界2（输出格式错乱重试）、模型降级兜底。

本模块职责：
    对上层（ReActAgent）屏蔽底层 SDK（OpenAI 兼容协议）细节，只暴露三个方法：
      1. invoke()             —— 普通对话，返回纯文本
      2. invoke_with_tools()  —— 带 Function Calling 的对话，返回文本 + 工具调用决策
      3. invoke_structured()  —— 要求模型返回 JSON，自动修复格式并兜底
    通用能力：全进程单例（只创建一个客户端）+ tenacity 自动重试 + 异常统一包装。

JavaScript 类比：
    相当于一个 api/llmClient.js 模块，内部 new OpenAI() 后封装三个请求函数，
    并加了「单例 + axios-retry + try/catch 统一抛错」三件套。
"""
import json
from typing import Optional, List, Dict, Any
# tenacity = Python 最常用的重试库：给函数加 @retry 装饰器，失败后自动重试
from tenacity import retry, stop_after_attempt, wait_exponential
from modules.common.logger import get_logger
from modules.common.exceptions import LLMException
# repair_json = 容错 JSON 解析器：模型吐出残缺/带 markdown 包裹的 JSON 时尽量修复
from modules.common.validator import repair_json
from config import get_settings

_logger = get_logger("llm")


class LLMClient:
    """DeepSeek Chat LLM 客户端（兼容 OpenAI 接口）。

    设计为单例：整个 Python 进程内无论 new 多少次 LLMClient()，拿到的都是同一个对象、
    复用同一个底层 HTTP 连接（类比 JS 模块里 export 一个全局唯一的 axios 实例）。
    """

    # 类属性：存放唯一实例的「槽位」，初始为 None（还没创建过）
    _instance = None

#     cls 就是类本身（class 的缩写），它是 new 方法的第一个参数，类似于 self 代表实例，cls 代表类。

# 简单对比
# 参数   代表什么   常见于
# self   当前实例对象   普通方法 def method(self)

# cls   当前类对象   类方法 @classmethod 和 new

# 在这段代码里的作用

# def new(cls):
#     if cls._instance is None:          # 从类上取 _instance 属性
#         cls._instance = super().new(cls)  # 创建实例并存到类上
#     return cls._instance               # 返回那个实例

# cls._instance → 访问类级别的属性 _instance（所有实例共享这一个）
# super().__new__(cls) → 告诉父类："请帮我创建一个 cls 这个类的实例"

# 为什么用 cls 而不是直接写类名？

# 因为 cls 是动态的。如果将来有子类继承 LLMClient：

# class SpecialLLMClient(LLMClient):
#     pass

# 调用 SpecialLLMClient() 时，cls 会自动变成 SpecialLLMClient，而不是写死的 LLMClient。这样代码更通用、更灵活。

# 💡 一句话总结：cls = "我自己这个类"，self = "我自己这个实例"。new 在实例还没创建出来的时候执行，所以没有 self，只能用 cls 来操作类本身。

# 这段代码实现的是单例模式，需要我帮你整理一份单例模式的速查表吗？包括线程安全写法、与 @lru_cache 的区别这些。

    def __new__(cls):
        """__new__ 负责「创建并返回实例」（在 __init__ 之前执行）。

        单例实现核心：
          - 第一次调用 LLMClient()：_instance 为 None → 真正创建一个对象并存进槽位
          - 之后再调用 LLMClient()：槽位已有对象 → 直接返回旧对象，不重新创建
        JS 没有 __new__ 这一层，等价写法：
            static getInstance() {
              if (!LLMClient._instance) LLMClient._instance = new LLMClient()
              return LLMClient._instance
            }
        """
        if cls._instance is None:
            # super().__new__(cls) = 调用父类 object 的 __new__，真正分配一个新对象
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化底层 SDK 客户端。

        注意：单例模式下每次 LLMClient() 都会再跑一次 __init__，
        所以用 hasattr(self, "_client") 做「只初始化一次」守卫：
        只有第一次（对象上还没有 _client 属性）才真正创建 OpenAI 客户端。
        """
        if not hasattr(self, "_client"):
            s = get_settings()  # 读取全局配置（api_key / base_url / 模型名）
            try:
                # 函数内延迟 import：避免模块加载阶段就强依赖 openai 包（也利于避免循环导入）
                from openai import OpenAI
                # DeepSeek 提供 OpenAI 兼容接口：只要改 base_url 和 api_key，SDK 原样复用
                self._client = OpenAI(
                    api_key=s.deepseek_api_key,
                    base_url=s.deepseek_base_url,
                )
                self._model: str = s.deepseek_model
                _logger.info(f"LLM客户端初始化: model={self._model}")
            except Exception as e:
                # 初始化失败（缺 key / 网络配置错误等）统一转成项目自定义异常向上抛
                raise LLMException("LLM客户端初始化失败", str(e))

    # @retry = tenacity 装饰器：函数抛异常时自动重试
    #   stop_after_attempt(2)   —— 最多尝试 2 次（首次 + 1 次重试）
    #   wait_exponential(min=1, max=5) —— 重试前等待 1~5 秒指数退避，避免疯狂打接口
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    def invoke(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.3, 
        max_tokens: int = 2048,
    ) -> str:
        """基础对话调用，返回文本。

        参数：
            prompt        用户输入内容
            system_prompt 系统提示词（设定 AI 角色/规则），可留空
            temperature   随机性：0 最确定，越高越发散
            max_tokens    回答最大长度（token 数）
        返回：
            模型回答的纯文本字符串（已 strip 去掉首尾空白）
        """
        messages = []
        # OpenAI 对话协议：messages 是一个有序列表，每个元素 {role, content}
        # system 消息必须排在最前面，用于设定全局行为
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        # user 消息 = 用户本轮提问
        messages.append({"role": "user", "content": prompt})
        try:
            # 真正发起 HTTP 请求：chat.completions.create 是 OpenAI 对话补全接口
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # 响应结构：resp.choices[0].message.content 即第一个候选回答的文本




        except Exception as e:
            _logger.error(f"LLM调用失败: {e}")
            # 不直接抛裸异常，包装成 LLMException 便于上层统一捕获
            raise LLMException("LLM调用失败", str(e))

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    def invoke_with_tools(
        self,
        messages: list,
        tools: list,
        temperature: float = 0.3,
    ) -> dict:
        """
        带工具的对话调用（原生 Function Calling）。

        【与 invoke() 的区别】
            invoke()         : 只发 prompt，拿一段文本回答 —— 适合"纯聊天"
            invoke_with_tools: 发完整 messages + 工具清单 (JSON Schema)，
                              让模型自己选"直接答" or "请求调工具" —— 适合 ReAct

        【返回值】{"content": 文本, "tool_calls": [...] 或 None}
            content    : 模型回的话（调工具时可能为空串）
            tool_calls : 模型想调的工具列表，没想调就是 None

        【两种决策结果对比】
            ✓ 模型决定调工具（用户问"3*7"）：
                返回 {
                    "content": "",
                    "tool_calls": [{
                        "name": "calculator",
                        "arguments": {"expression": "3*7"},
                        "id": "call_abc123"
                    }]
                }
                → Agent 拿 tool_calls 去 act 节点执行 calculator，结果回 reason

            ✗ 模型决定直接答（用户问"你好"）：
                返回 {
                    "content": "你好！有什么可以帮你的？",
                    "tool_calls": None
                }
                → Agent 看 tool_calls 为 None，直接把 content 当 final_answer 收尾

        【为什么 tool_choice="auto"？】项目硬约束：工具调用必须由 LLM 自主决策，
            不强制调工具。auto 让模型基于上下文判断"现在该不该用工具"。
        """
        breakpoint()  # 【调试断点】LLM调用入口：可看 messages / tools（发给模型的内容）
        try:
            # create() = 向 /chat/completions 发一次 HTTP 请求（本文件真正联网调模型的一下），
            # 括号内是随请求发出去的配置，各司其职：
            resp = self._client.chat.completions.create(
                model=self._model,  # 用哪个模型（deepseek-chat）
                messages=messages,  # 问什么（system+user 对话内容，支撑多轮 ReAct）
                # tools=None 时不能传空列表，否则部分兼容接口报错，这里做兼容处理
                tools=tools if tools else None,  # 给模型的工具清单（JSON Schema）
                # tool_choice="auto" = 把「要不要调工具、调哪个」的决策权完全交给模型
                # （项目硬约束：工具调用必须由 LLM 自主决策）
                tool_choice="auto" if tools else None,  # 让模型自主决定调不调工具
                # temperature：回答的随机性，0 最确定、越高越发散；
                # Agent 工具决策要稳定，默认传 0.3
                temperature=temperature,
                # max_tokens：回答长度上限（token 数），到数强制截断；
                # 工具参数 JSON 可能较长，给 4096 防止生成到一半被截断
                max_tokens=4096,  # 工具参数可能较长，给足 token
            )
            msg = resp.choices[0].message
            # content 可能为 None（模型选择只调工具、不说话），用 or "" 兜底成空串
            content = msg.content or ""
            tool_calls = None
            # 模型请求调用工具时，msg.tool_calls 是一个数组（一次可请求多个工具）
            if msg.tool_calls:
                tool_calls = []
                for tc in msg.tool_calls:
                    # 注意：SDK 返回的工具参数是 JSON 字符串，不是 dict，需要解析
                    args_str = tc.function.arguments
                    # 边界2：Function Call参数JSON修复
                    # 模型可能吐出非法 JSON（尾逗号/缺括号/带多余文本），repair_json 尽力修复
                    args = repair_json(args_str)
                    if args is None:
                        # 修复也失败则给空 dict，保证下游取参数时不崩（相当于默认无参调用）
                        args = {}
                    tool_calls.append({
                        "name": tc.function.name,   # 工具名，对应 ToolRegistry 里注册的名字
                        "arguments": args,          # 已解析成 dict 的工具入参
                        "id": tc.id,                # 本次工具调用的唯一ID（多轮对话回填结果时要用）
                    })
                _logger.info(f"LLM请求工具调用: {[tc['name'] for tc in tool_calls]}")
            breakpoint()  # 【调试断点】LLM返回结果：可看 content / tool_calls（模型决策）
            return {"content": content, "tool_calls": tool_calls}
        except Exception as e:
            _logger.error(f"LLM工具调用失败: {e}")
            raise LLMException("LLM工具调用失败", str(e))

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    def invoke_structured(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> dict:
        """
        结构化输出调用：要求LLM返回JSON格式。
        自动修复格式，失败重试2次，最终兜底返回 {error}。

        典型场景：生成摘要、信息抽取、意图分类等需要程序直接读取结果（而非展示给人）的调用。
        temperature 默认 0.2 更低，保证输出稳定。
        """
        # 拼 system + user 两段提示词
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        # 显式追加输出要求，双保险约束模型只吐 JSON
        full_prompt += '\n\n请严格以JSON格式输出，不要包含额外文本。'
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=temperature,
                max_tokens=2048,
                # response_format 是 OpenAI 的「JSON 模式」开关，
                # 强制模型输出必须是合法 JSON 对象（不兼容的服务端会抛异常，走下方降级）
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content.strip()
            # 即便开了 JSON 模式，仍容错解析一次
            result = repair_json(content)
            if result is None:
                # 解析失败：记录原始内容（截断前500字符防日志爆量），返回错误 dict 而非抛异常
                _logger.error(f"结构化输出解析失败: {content[:200]}")
                return {"error": "格式解析失败", "raw": content[:500]}
            return result
        except Exception as e:
            # 降级：不支持 response_format 时用普通调用
            # 某些兼容接口不识别 response_format 参数 → 退回 invoke() 普通对话，
            # 仅靠提示词 + repair_json 保证拿到 JSON（能力降级但不中断业务）
            _logger.warning(f"结构化调用降级: {e}")
            content = self.invoke(prompt, system_prompt, temperature)
            result = repair_json(content)
            return result if result else {"error": "格式解析失败", "raw": content[:500]}
