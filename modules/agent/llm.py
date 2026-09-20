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
from tenacity import retry, stop_after_attempt, wait_exponential
from modules.common.logger import get_logger
from modules.common.exceptions import LLMException
from modules.common.validator import repair_json
from config import get_settings

_logger = get_logger("llm")


class LLMClient:
    """DeepSeek Chat LLM 客户端（兼容 OpenAI 接口）。

    设计为单例：整个 Python 进程内无论 new 多少次 LLMClient()，拿到的都是同一个对象、
    复用同一个底层 HTTP 连接（类比 JS 模块里 export 一个全局唯一的 axios 实例）。
    """

    _instance = None













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
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化底层 SDK 客户端。

        注意：单例模式下每次 LLMClient() 都会再跑一次 __init__，
        所以用 hasattr(self, "_client") 做「只初始化一次」守卫：
        只有第一次（对象上还没有 _client 属性）才真正创建 OpenAI 客户端。
        """
        if not hasattr(self, "_client"):
            s = get_settings()
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=s.deepseek_api_key,
                    base_url=s.deepseek_base_url,
                )
                self._model: str = s.deepseek_model
                _logger.info(f"LLM客户端初始化: model={self._model}")
            except Exception as e:
                raise LLMException("LLM客户端初始化失败", str(e))

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
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )




        except Exception as e:
            _logger.error(f"LLM调用失败: {e}")
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
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
                temperature=temperature,
                max_tokens=4096,
            )
            msg = resp.choices[0].message
            content = msg.content or ""
            tool_calls = None
            if msg.tool_calls:
                tool_calls = []
                for tc in msg.tool_calls:
                    args_str = tc.function.arguments
                    args = repair_json(args_str)
                    if args is None:
                        args = {}
                    tool_calls.append({
                        "name": tc.function.name,
                        "arguments": args,
                        "id": tc.id,
                    })
                _logger.info(f"LLM请求工具调用: {[tc['name'] for tc in tool_calls]}")
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
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        full_prompt += '\n\n请严格以JSON格式输出，不要包含额外文本。'
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=temperature,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content.strip()
            result = repair_json(content)
            if result is None:
                _logger.error(f"结构化输出解析失败: {content[:200]}")
                return {"error": "格式解析失败", "raw": content[:500]}
            return result
        except Exception as e:
            _logger.warning(f"结构化调用降级: {e}")
            content = self.invoke(prompt, system_prompt, temperature)
            result = repair_json(content)
            return result if result else {"error": "格式解析失败", "raw": content[:500]}
