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
from typing import Optional

from modules.common.logger import get_logger
from modules.multimodal.media import MultiModalInput
from config import get_settings

_logger = get_logger("vision")


class VisionClient:
    """Vision 模型客户端（OpenAI 兼容协议）。"""

    def __init__(self, client=None):
        self._client = client
        self._inited = client is not None

    def _ensure_client(self):
        """延迟建客户端：没配 Key 时连 SDK 都不 import，避免无谓依赖。"""
        if self._inited:
            return self._client
        self._inited = True
        s = get_settings()
        if not (s.vision_enabled and s.vision_api_key):
            self._client = None
            return None
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=s.vision_api_key, base_url=s.vision_base_url)
            _logger.info(f"视觉客户端初始化: model={s.vision_model}")
        except Exception as e:
            _logger.warning(f"视觉客户端初始化失败（走文本降级）: {e}")
            self._client = None
        return self._client

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
        s = get_settings()
        if mm.image_count == 0:
            return self._fallback_text(mm, system_prompt, reason="无图片，按纯文本处理")

        client = self._ensure_client()
        if client is None:
            return self._fallback_text(mm, system_prompt, reason="视觉模型未启用，图片降级为占位文本")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": mm.to_openai_content()})
        try:
            resp = client.chat.completions.create(
                model=s.vision_model,
                messages=messages,
                temperature=s.vision_temperature,
                max_tokens=s.vision_max_tokens,
                timeout=s.vision_timeout_seconds,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            _logger.error(f"视觉调用失败（转文本降级）: {e}")
            return self._fallback_text(mm, system_prompt, reason=f"视觉调用失败: {e}")

    def _fallback_text(self, mm: MultiModalInput, system_prompt: str, reason: str) -> str:
        """统一降级：用 text_only 视图走文本模型，文本模型再失败则返回空串。"""
        _logger.warning(f"多模态降级：{reason}")
        try:
            from modules.agent.llm import LLMClient
            return LLMClient().invoke(mm.text_only, system_prompt=system_prompt)
        except Exception as e:
            _logger.error(f"文本降级也失败: {e}")
            return ""


_VISION_CLIENT: Optional[VisionClient] = None


def get_vision_client() -> VisionClient:
    """全局单例（复用同一个 HTTP 连接池）。"""
    global _VISION_CLIENT
    if _VISION_CLIENT is None:
        _VISION_CLIENT = VisionClient()
    return _VISION_CLIENT
