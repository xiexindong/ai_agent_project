"""
视觉（多模态）调用模块
======================
把「文本 + 图片」的多模态输入发给支持 Vision 的 OpenAI 兼容模型。

【为什么单独开一个客户端，不复用 modules/agent/llm.py 的 LLMClient】
    LLMClient 面向的是 DeepSeek 文本模型：invoke() 只吃 str，
    messages 里没有 image_url 的位置。两个模型的能力、地址、Key 都不同
    （文本走 DeepSeek，视觉走 OpenAI/智谱 GLM-4V 之类），
    强行塞进同一个类会让配置项互相污染。这里独立一份，接口保持一致。

【降级链】任何一环失败都不让请求 500：
    视觉模型不可用 → 退回文本模型，把图片降级成占位文本 → 再失败 → 返回空串
"""
from typing import Optional  # 从 typing 导入 Optional，用于声明可空类型注解

from modules.common.logger import get_logger  # 从项目日志模块导入 get_logger，用于获取模块级日志记录器
from modules.multimodal.media import MultiModalInput  # 从同级 media 模块导入 MultiModalInput，作为多模态输入的数据结构
from config import get_settings  # 从项目配置模块导入 get_settings，用于读取视觉模型相关配置

_logger = get_logger("vision")  # 创建模块级日志记录器，命名为 "vision"，用于记录视觉客户端的运行日志


# 【什么】Vision 模型客户端（进程内单例）
# 【作用】把多模态输入发给支持 Vision 的模型，拿不到就降级到文本链路
# 【为什么】视觉模型与文本模型在协议、配置、能力上差异大，需要独立客户端封装，同时提供稳定的降级策略
class VisionClient:
    """Vision 模型客户端（OpenAI 兼容协议）。"""

    # 【什么】构造时可注入外部 OpenAI 客户端
    # 【作用】注入了就直接用（测试换 Mock）；没注入则等首次调用时再惰性创建
    # 【为什么】依赖注入便于单元测试时替换 Mock 对象；惰性创建避免无 Key 时提前初始化失败
    def __init__(self, client=None):
        self._client = client            # 底层 OpenAI SDK 客户端，None 表示未启用或未创建  # 存储底层 OpenAI SDK 客户端实例，初始为传入值或 None
        self._inited = client is not None  # True 表示已尝试过初始化，不再重复判空  # 标记是否已尝试过初始化：传入了客户端则视为已初始化，否则等待首次调用时惰性创建

    # 【什么】惰性创建 OpenAI 客户端
    # 【作用】没配视觉 Key 时不 import SDK；创建失败只告警，返回 None 走降级
    # 【为什么】避免无配置时引入无谓依赖；初始化失败不应导致程序崩溃，应优雅降级到文本链路
    def _ensure_client(self):
        """延迟建客户端：没配 Key 时连 SDK 都不 import，避免无谓依赖。"""
        if self._inited:  # 如果已经尝试过初始化（无论成功或失败）
            return self._client  # 直接返回已有客户端实例（可能是 None），避免重复创建
        self._inited = True  # 标记已尝试过初始化，后续不再重复执行
        s = get_settings()  # 获取全局配置对象
        if not (s.vision_enabled and s.vision_api_key):  # 如果视觉功能未启用或未配置 API Key
            self._client = None  # 将客户端置为 None，表示视觉链路不可用
            return None  # 返回 None，调用方应走降级逻辑
        try:
            from openai import OpenAI  # 延迟导入 OpenAI SDK，避免未配置时引入依赖
            self._client = OpenAI(api_key=s.vision_api_key, base_url=s.vision_base_url)  # 使用配置的 Key 和 Base URL 创建 OpenAI 客户端
            _logger.info(f"视觉客户端初始化: model={s.vision_model}")  # 记录初始化成功日志，包含模型名称
        except Exception as e:  # 捕获所有异常（如网络问题、SDK 未安装、配置错误等）
            _logger.warning(f"视觉客户端初始化失败（走文本降级）: {e}")  # 记录警告日志，说明初始化失败及原因
            self._client = None  # 初始化失败，将客户端置为 None，触发降级
        return self._client  # 返回创建好的客户端实例（成功）或 None（失败）

    # 【什么】多模态调用主入口
    # 【作用】有图且视觉可用时走 Vision；否则一路降级到文本模型，绝不抛异常
    # 【为什么】多模态调用的稳定性至关重要，任何环节失败都要有兜底策略，保证服务不中断
    def invoke(self, mm: MultiModalInput, system_prompt: str = "") -> str:
        """
        多模态调用。

        【输入 → 输出对照】
            输入: MultiModalInput(instruction="图里销售额最高的月份？",
                                 parts=[ImagePart(source="chart.png")])
            输出: "由图可见，销售额最高的是 11 月，约 480 万。"

        【降级路径】
            没有图片      → 直接走文本模型（LLMClient.invoke）
            视觉模型未配置 → 把图片降级成 [图片] 占位文本，仍走文本模型
            调用抛异常     → 记录日志并返回空串，绝不向上抛（调用方自行兜底）
        """
        s = get_settings()  # 获取全局配置对象
        if mm.image_count == 0:  # 如果输入中没有图片
            return self._fallback_text(mm, system_prompt, reason="无图片，按纯文本处理")  # 直接走文本降级通道，按纯文本处理

        client = self._ensure_client()  # 惰性获取或创建视觉客户端
        if client is None:  # 如果客户端不可用（未配置或初始化失败）
            return self._fallback_text(mm, system_prompt, reason="视觉模型未启用，图片降级为占位文本")  # 走文本降级通道，图片转为占位文本

        messages = []  # 初始化 OpenAI 消息列表
        if system_prompt:  # 如果有系统提示词
            messages.append({"role": "system", "content": system_prompt})  # 将系统提示词作为 system 角色消息加入列表
        messages.append({"role": "user", "content": mm.to_openai_content()})  # 将多模态输入转换为 OpenAI 格式后作为 user 角色消息加入列表
        try:
            resp = client.chat.completions.create(  # 调用 OpenAI 兼容的 chat completions API
                model=s.vision_model,  # 指定视觉模型名称（从配置读取）
                messages=messages,  # 传入组装好的消息列表
                temperature=s.vision_temperature,  # 指定采样温度（从配置读取）
                max_tokens=s.vision_max_tokens,  # 指定最大生成 token 数（从配置读取）
                timeout=s.vision_timeout_seconds,  # 指定请求超时时间（从配置读取）
            )
            return (resp.choices[0].message.content or "").strip()  # 提取响应文本：取第一个 choice 的 content，None 转空串，去除首尾空白
        except Exception as e:  # 捕获所有异常（如网络超时、API 错误、配额不足等）
            _logger.error(f"视觉调用失败（转文本降级）: {e}")  # 记录错误日志，说明视觉调用失败及原因
            return self._fallback_text(mm, system_prompt, reason=f"视觉调用失败: {e}")  # 走文本降级通道，携带失败原因

    # 【什么】统一的文本降级通道
    # 【作用】用纯文本视图再问一次文本模型；文本模型也失败则返回空串兜底
    # 【为什么】视觉链路失败时，文本模型是最后兜底；文本模型再失败也不能让异常扩散到上游
    def _fallback_text(self, mm: MultiModalInput, system_prompt: str, reason: str) -> str:
        """统一降级：用 text_only 视图走文本模型，文本模型再失败则返回空串。"""
        _logger.warning(f"多模态降级：{reason}")  # 记录警告日志，说明发生了降级及原因
        try:
            from modules.agent.llm import LLMClient  # 延迟导入文本模型客户端，避免循环依赖
            return LLMClient().invoke(mm.text_only, system_prompt=system_prompt)  # 创建文本客户端实例，用纯文本视图调用文本模型
        except Exception as e:  # 捕获所有异常（如文本模型也未配置、网络故障等）
            _logger.error(f"文本降级也失败: {e}")  # 记录错误日志，说明文本降级也失败及原因
            return ""  # 返回空串作为最终兜底，保证调用方不会收到异常


_VISION_CLIENT: Optional[VisionClient] = None   # 进程级单例实例，首次调用时创建  # 声明模块级私有变量，存储 VisionClient 单例，初始为 None


# 【什么】获取全局唯一的 VisionClient
# 【作用】复用同一个实例（含 HTTP 连接池），避免每次请求重复建连
# 【为什么】HTTP 客户端创建开销大（TCP 连接、TLS 握手），单例模式可复用连接池，提升性能
def get_vision_client() -> VisionClient:
    """全局单例（复用同一个 HTTP 连接池）。"""
    global _VISION_CLIENT  # 声明使用模块级全局变量
    if _VISION_CLIENT is None:  # 如果单例尚未创建
        _VISION_CLIENT = VisionClient()  # 创建 VisionClient 实例并赋值给全局变量
    return _VISION_CLIENT  # 返回全局唯一的 VisionClient 实例
