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
import time
from typing import Optional, List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from modules.common.logger import get_logger
from modules.common.exceptions import LLMException
from modules.common.validator import repair_json
from config import get_settings
from modules.common.circuit_breaker import CircuitBreaker

_logger = get_logger("llm")

# 不可降级的 HTTP 状态码：这些是「请求本身有问题」，换个模型结果一模一样。
#   400 参数错 / 401 认证失败 / 402 欠费 / 403 无权限 / 404 模型不存在 / 405 / 409 / 413 / 422
# 反过来看：无状态码（网络层超时、连接失败）与 429 限流、5xx 服务端故障，都是「可降级」。
_NO_DEGRADE_STATUS = {400, 401, 402, 403, 404, 405, 409, 413, 422}


def _is_degradable(exc: BaseException) -> bool:
    """这个异常是否值得「换个模型再试一次」。

    True  ：超时 / 连接失败 / 429 / 5xx —— 下游瞬时故障，降级有意义
    False ：400 / 401 / 402 / 403 / 404 / 422 —— 请求本身的错，换谁都一样；
            这类既不该降级（白等一次超时），也不该计入熔断
            （否则一次参数 bug 就能把主模型熔断掉，整条链路跟着被误伤）

    判据用「HTTP 状态码」而不是异常类名：OpenAI SDK 的网络异常
    （APITimeoutError / APIConnectionError）没有 status_code，
    所以「无状态码」就等于「请求没到达服务端」，一律视为可降级。
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        return True
    return status not in _NO_DEGRADE_STATUS


def _llm_retry():
    return retry(
        stop=stop_after_attempt(max(1, get_settings().llm_max_attempts)),
        wait=wait_exponential(min=1, max=8),
    )


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
                self.usage_context: dict = {}
                # 熔断器跟着单例走：整个进程共用一个，内部按模型名隔离计数
                self._breaker = CircuitBreaker(
                    failure_threshold=s.cb_failure_threshold,
                    recovery_timeout=s.cb_recovery_seconds,
                )
                _logger.info(f"LLM客户端初始化: model={self._model}")
            except Exception as e:
                raise LLMException("LLM客户端初始化失败", str(e))

    def set_usage_context(
        self,
        user_code: str = "",
        dept_code: str = "",
        session_id: str = "",
        task_id: str = "",
    ):
        """设置本次调用的归属上下文，供 token 用量按人/部门归集。

        task_id 是跨库对账的主键：PG 的 chat_history 与 MySQL 的 sys_llm_usage
        靠它指向同一轮对话，缺省时从请求上下文（modules.common.context）取。
        """
        self.usage_context = {
            "user_code": user_code,
            "dept_code": dept_code,
            "session_id": session_id,
            "task_id": task_id,
        }

    def _chat(self, **kwargs):
        """统一出口：熔断快速失败 + 备用模型降级 + token 用量落库。"""
        s = get_settings()
        started = time.time()
        # ------------------------------------------------------------------
        # 【model 必须由这里兜底传入】
        #   上层三个 invoke* 方法都只传 messages/temperature，没人传 model，
        #   OpenAI SDK 新版会在入口强校验并抛：
        #     Missing required arguments; Expected either ('messages' and 'model')...
        #
        # 【不这么写会怎样】
        #   每次调用都失败 → tenacity 重试耗尽（默认 3 次、指数退避 1~8 秒）
        #   → 单次对话耗时被拉到 3~4 秒且答案为空。接口仍返回 200，
        #     所以只看 HTTP 状态码的压测会显示"全部成功"，是最难发现的静默故障。
        #
        # 【为什么从 setdefault 改成 pop】model 不再原样转发给 SDK，
        #   而是取出来当「主模型」用 —— 熔断计数、降级判断都按它做；
        #   真正发给 SDK 的 model 由下面的 _request() 显式指定。
        # ------------------------------------------------------------------
        primary = kwargs.pop("model", None) or self._model
        fb = s.fallback_model
        can_fallback = bool(fb) and fb != primary

        # ------------------------------------------------------------------
        # _chat 是 invoke / invoke_with_tools / invoke_structured 的唯一出口，
        # 一次调用要过两道闸：
        #
        #   闸 1 · 熔断：主模型连续失败已达阈值 → 试都不试，直接走备用模型。
        #          省下的正是「注定失败的那次等待」—— 默认 30s 超时。
        #   闸 2 · 降级：主模型这次真的失败了 → 换备用模型重发一次。
        #          只对「换个下游可能就好」的故障降级：超时/连接失败/429/5xx；
        #          400/401/402 这类请求本身的错直接抛 —— 换谁都一样，且不计入熔断。
        #
        #   处理前 kwargs（model 已被上一行取出，其余由上层 invoke* 传入）：
        #       {"messages": [{"role": "user", "content": "3*7等于几"}],
        #        "temperature": 0.3, "max_tokens": 2048}
        #   处理后 resp —— OpenAI SDK 的 ChatCompletion 对象，回答不在顶层：
        #       resp.choices[0].message.content → "3*7=21"   ← 正文，invoke 里取的就是它
        #       resp.model                      → 实际服务的模型，计费按它算
        #       resp.usage.prompt_tokens        → 12          ← _log_usage 取它
        #
        # 【有熔断器之后的耗时差别】主备都故障时（llm_max_attempts=3）：
        #   只有降级：每次重试都先撞一次主模型 → 3 × (30s + 30s) ≈ 3 分钟
        #   加了熔断：熔断期间直接跳过主模型 → 3 × 30s ≈ 1.5 分钟
        # ------------------------------------------------------------------
        if can_fallback and not self._breaker.allow(primary):
            # 熔断中：不做无谓尝试，直接降级。这一步是「省时间」而不是「容错」
            _logger.warning(f"熔断器打开，跳过主模型 {primary}，直接使用备用模型 {fb}")
            resp = self._request(fb, s, **kwargs)
        else:
            try:
                resp = self._request(primary, s, **kwargs)
                self._breaker.record_success(primary)
            except Exception as e:
                if not _is_degradable(e):
                    raise                       # 请求本身的错：降级无意义，也不记熔断
                self._breaker.record_failure(primary)
                if not can_fallback:
                    raise
                _logger.warning(f"主模型 {primary} 调用失败，降级备用模型 {fb}: {e}")
                resp = self._request(fb, s, **kwargs)

        # ------------------------------------------------------------------
        # 【这行在干什么】调用成功后记一笔用量流水，用来核对费用。
        #   一次对话 = sys_llm_usage 里一行：谁（user_code）、哪个部门（dept_code）、
        #   哪轮对话（session_id / task_id）、用了哪个模型、烧了多少 token、等了多久。
        #   三个 invoke* 都从这里出去，主模型和降级后的备用模型也都从这里出去，
        #   所以调用路径再多也不会漏账。
        #
        # 【处理前 → 处理后】
        #   处理前 resp 由服务端带回用量，started 是进入 _chat 的时点：
        #       resp.model                    → "deepseek-chat"   ← 实际服务的模型
        #       resp.usage.prompt_tokens      → 12
        #       resp.usage.completion_tokens  → 5
        #       time.time() - started         → 0.82 秒
        #   处理后 MySQL 多一行（total_tokens 由两列相加）：
        #       sys_llm_usage(user_code='U001', dept_code='D_TECH',
        #                     session_id='sess-7f3a', task_id='T-20260921-3f9c1a2b',
        #                     model='deepseek-chat',
        #                     prompt_tokens=12, completion_tokens=5, total_tokens=17,
        #                     latency_ms=820)
        #
        # 【核对费用的口径】按部门：GROUP BY dept_code 后 SUM(total_tokens)；
        #   按人则换成 user_code（idx_user_time 索引）。model 列要留着算价：
        #   主模型与备用模型单价不同，混在一起就不准，所以记的是 resp.model
        #   （实际服务的那一个），而不是我们请求时点名的那个。
        #
        # 【latency_ms 为什么在这里算】SDK 不返回端到端耗时。started 取自 _chat 入口，
        #   包含了降级重发那次，记的是用户真实感受到的等待，与「主备都挂最坏 3 分钟」
        #   的止损口径一致。
        #
        # 【两种记不上账的边界（都不影响对话返回）】
        #   1. resp.usage 为空（部分兼容接口不返回）或 RBAC_ENABLED=false
        #      → _log_usage 开头直接 return，不写库；
        #   2. MySQL 写失败 → _log_usage 内只留一条 debug 日志，异常不上抛。
        #      记账是旁路，不能反过来把主链路拖挂。
        # ------------------------------------------------------------------
        self._log_usage(resp, latency_ms=int((time.time() - started) * 1000))
        return resp

    def _request(self, model: str, s, **kwargs):
        """按指定模型发一次请求 —— 本类唯一真正发起 HTTP 的地方。

        model 由调用方显式传入（主模型或备用模型），熔断与降级共用这一个出口，
        不会出现「两处 HTTP 代码各写一遍、改一处漏一处」的漂移。
        """
        return self._client.chat.completions.create(
            model=model, timeout=s.llm_timeout_seconds, **kwargs
        )

    def _log_usage(self, resp, latency_ms: int):
        usage = getattr(resp, "usage", None)
        if not usage or not get_settings().rbac_enabled:
            return
        try:
            from modules.enterprise.audit import log_llm_usage
            from modules.common import context as req_ctx
            ctx = getattr(self, "usage_context", {}) or {}
            log_llm_usage(
                user_code=ctx.get("user_code", ""),
                model=getattr(resp, "model", self._model),
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                latency_ms=latency_ms,
                dept_code=ctx.get("dept_code", ""),
                session_id=ctx.get("session_id", ""),
                task_id=ctx.get("task_id", "") or req_ctx.get_task_id(),
            )
        except Exception as e:
            _logger.debug(f"用量上报失败: {e}")

    @_llm_retry()
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
            resp = self._chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()




        except Exception as e:
            _logger.error(f"LLM调用失败: {e}")
            raise LLMException("LLM调用失败", str(e))

    @_llm_retry()
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
            resp = self._chat(
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

    @_llm_retry()
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
            resp = self._chat(
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
