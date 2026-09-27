"""
多模态内容模块
==============
把「一段对话输入」统一抽象成若干内容片段（ContentPart）：
文本、图片、文件，用 pydantic v2 判别联合（discriminated union）收口。

本模块只做三件事：
    1. 定义数据形状（TextPart / ImagePart / FilePart）
    2. 在实例化那一刻做校验与归一（格式白名单、体积上限、路径→base64）
    3. 转成 OpenAI Vision 协议要求的 messages content 数组

【为什么用 pydantic 而不是裸 dict】
    多模态最容易出的事故是「形状对了、内容错了」：
    图片传成 http 链接却按本地路径读、pdf 被当图片塞进 image_url、
    5MB 的图直接发给模型导致 413。这些错误都不会在 Python 层报错，
    而是在远端花掉一次调用后才暴露。用 pydantic 在入口就 validation 掉。

【为什么用判别联合（discriminator）而不是 Union 顺序匹配】
    Union[TextPart, ImagePart, FilePart] 在 pydantic v2 里默认"聪明模式"，
    会按顺序尝试，一个字段相似的 dict 可能被错误匹配到前一个类型。
    加 kind: Literal["text"] 做判别标签后，匹配只认 kind 字段，
    不会因为字段长得像而匹配错（这是 v2 相比 v1 的关键改进）。
"""
import base64
import mimetypes
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator

from config import get_settings

# 【格式白名单】只放行模型侧确认能解的图；不改这一份，别在别处硬编码
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
# 【文本文件白名单】pdf 走可选提取，其余按纯文本读
ALLOWED_FILE_SUFFIX = {".txt", ".md", ".csv", ".json", ".log", ".pdf"}


# 【什么】内容片段的公共基类
# 【作用】统一声明 kind 判别字段，并定下「降级为纯文本」的默认行为
class BasePart(BaseModel):
    """内容片段基类。kind 字段是判别联合的标签，子类各自写死字面量。"""

    kind: str = Field(description="片段类型判别标签：text / image / file")

    # 【什么】把本片段降级为纯文本
    # 【作用】给不支持视觉的模型一个兜底视图；基类返回空串，由子类覆写
    def to_text(self) -> str:
        """降级为纯文本（给不支持视觉的模型用）。子类覆写。"""
        return ""


# 【什么】纯文本片段
# 【作用】承载文字内容，无归一化动作，是判别联合里的 text 分支
class TextPart(BasePart):
    """文本片段。"""

    kind: Literal["text"] = "text"   # 判别标签，固定为 "text"
    text: str = Field(description="文本内容")

    # 【什么】文本片段的降级视图
    # 【作用】原样返回 text，本身就是文本无需加工
    def to_text(self) -> str:
        return self.text


# 【什么】图片片段
# 【作用】接收 data URL / 远程 URL / 本地路径三种来源，归一成可直接发送的图片数据
class ImagePart(BasePart):
    """图片片段。source 三种形态都收：data URL / http(s) URL / 本地文件路径。

    【输入 → 输出对照】
        输入: ImagePart(source="chart.png")            # 本地相对路径
        归一: data_url="data:image/png;base64,iVBOR..."  # 读文件后自动转 base64
              bytes_size=18234  media_type="image/png"

        输入: ImagePart(source="https://x.com/a.png")
        归一: data_url="https://x.com/a.png"             # 远程图不下载，原样透传
              bytes_size=0

        输入: ImagePart(source="data:image/gif;base64,R0lGOD...")
        归一: bytes_size=42  media_type="image/gif"      # 只校验，不重复编码
    """

    kind: Literal["image"] = "image"
    source: str = Field(description="data URL / http(s) URL / 本地文件路径")
    caption: str = Field(default="", description="图片说明（可选，会作为文本一起发给模型）")
    media_type: str = Field(default="", description="归一化后填：image/png 等")
    data_url: str = Field(default="", description="归一化后可直接发给模型的 URL")
    bytes_size: int = Field(default=0, description="原始字节数，用于体积校验")

    # 【什么】实例化后自动执行的归一化校验器
    # 【作用】把 source 的三种形态统一成可直接发送的数据，并卡住类型与体积
    @model_validator(mode="after")
    def _normalize(self) -> "ImagePart":
        # ------------------------------------------------------------------
        # 【案例1：远程 URL】不下载。下载要引入网络依赖 + 超时 + SSRF 风险，
        #   交给模型侧自己去取；我们只做协议头校验。
        # ------------------------------------------------------------------
        if self.source.startswith(("http://", "https://")):
            self.data_url = self.source
            return self

        # ------------------------------------------------------------------
        # 【案例2：data URL】拆出 media_type 与 base64 载荷，只校验不重编码。
        #   不这么写的后果：外层再包一层 data:image/png;base64, 前缀，
        #   模型收到双重前缀直接报 400 invalid image。
        # ------------------------------------------------------------------
        if self.source.startswith("data:"):
            header, _, payload = self.source.partition(",")
            if ";base64" not in header or not payload:
                raise ValueError("data URL 格式非法，应为 data:<type>;base64,<payload>")
            self.media_type = header[len("data:"):].split(";")[0]
            self.data_url = self.source
            self.bytes_size = self._decoded_size(payload)
        else:
            # ------------------------------------------------------------------
            # 【案例3：本地路径】读盘 → 猜 MIME → 转 base64 → 组装 data URL。
            #   不这么写的后果：模型服务在内网/Docker 里根本访问不到你的本地路径。
            # ------------------------------------------------------------------
            path = Path(self.source)
            if not path.exists():
                raise ValueError(f"图片文件不存在: {self.source}")
            raw = path.read_bytes()
            self.bytes_size = len(raw)
            guess = mimetypes.guess_type(path.name)[0] or ""
            self.media_type = guess if guess in ALLOWED_IMAGE_TYPES else ""
            self.data_url = (
                f"data:{self.media_type};base64,{base64.b64encode(raw).decode('ascii')}"
            )

        if self.media_type and self.media_type not in ALLOWED_IMAGE_TYPES:
            raise ValueError(
                f"不支持的图片类型 {self.media_type}，允许: {sorted(ALLOWED_IMAGE_TYPES)}"
            )

        limit_mb = get_settings().vision_max_image_mb
        if self.bytes_size and self.bytes_size > limit_mb * 1024 * 1024:
            raise ValueError(
                f"图片 {self.bytes_size / 1048576:.2f}MB 超过上限 {limit_mb}MB"
            )
        return self

    # 【什么】按 base64 载荷反算图片的原始字节数
    # 【作用】给体积上限校验提供数据；解码失败返回 0（0 表示跳过体积校验）
    @staticmethod
    def _decoded_size(payload: str) -> int:
        """算 base64 载荷解码后的真实字节数（校验失败返回 0，不抛异常）。"""
        try:
            return len(base64.b64decode(payload, validate=False))
        except Exception:
            return 0

    # 【什么】图片片段降级成占位文本
    # 【作用】带 caption 时输出「[图片：说明]」，无 caption 时输出「[图片]」
    def to_text(self) -> str:
        return f"[图片{self.caption and '：' + self.caption}]"


# 【什么】文件片段
# 【作用】在实例化时就把文件读成文本：文本文件直接读，pdf 走可选抽取
class FilePart(BasePart):
    """文件片段：文本文件直接读成字符串；pdf 尝试用 pypdf 抽文本，抽不到降级占位。"""

    kind: Literal["file"] = "file"
    path: str = Field(description="本地文件路径")
    name: str = Field(default="", description="展示用文件名")
    content: str = Field(default="", description="抽取出的文本（截断到 max_file_chars）")
    bytes_size: int = Field(default=0)

    # 【什么】实例化后立即读盘并抽取文本
    # 【作用】保证 downstream 拿到的 FilePart 一定带 content，无需再判空或处理 IO
    @model_validator(mode="after")
    def _load(self) -> "FilePart":
        path = Path(self.path)
        if not path.exists():
            raise ValueError(f"文件不存在: {self.path}")
        if path.suffix.lower() not in ALLOWED_FILE_SUFFIX:
            raise ValueError(f"不支持的文件类型 {path.suffix}，允许: {sorted(ALLOWED_FILE_SUFFIX)}")
        self.name = self.name or path.name
        self.bytes_size = path.stat().st_size
        self.content = self._extract(path)
        return self

    # 【什么】按后缀抽取文件正文
    # 【作用】pdf 走 pypdf 抽文本，其余按 utf-8 读文本；两种情况都截断到 limit 字
    @staticmethod
    def _extract(path: Path) -> str:
        limit = get_settings().vision_max_file_chars   # 文本抽取上限（字），来自配置 vision_max_file_chars
        if path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
                text = "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
                return text[:limit]
            except Exception:
                # pdf 抽取失败不阻断主流程：留占位，让 LLM 知道"有这个文件但没读到"
                return f"（PDF 文本抽取失败：{path.name}）"
        try:
            return path.read_text(encoding="utf-8", errors="ignore")[:limit]
        except Exception:
            return f"（文件读取失败：{path.name}）"

    # 【什么】文件片段降级成「文件名 + 正文」文本
    # 【作用】不支持文件的模型也能看到文件名与已抽取的内容
    def to_text(self) -> str:
        return f"[文件 {self.name}]\n{self.content}"


# 【判别联合】靠 kind 字段选类型，避免字段相似导致的错误匹配
ContentPart = Annotated[
    Union[TextPart, ImagePart, FilePart],
    Field(discriminator="kind"),
]


# 【什么】一次完整的多模态输入
# 【作用】把「指令 + 若干内容片段」打包，并负责整体限额校验与协议转换
# 【什么】一次完整的多模态输入
# 【作用】把若干内容片段与一条指令打包，并提供转 OpenAI 协议与纯文本两种视图
class MultiModalInput(BaseModel):
    """一次多模态输入 = 若干片段 + 一条指令。

    【输入 → 输出对照】
        输入: MultiModalInput(instruction="这张图里有什么？",
                             parts=[TextPart(text="图上数字念给我"),
                                    ImagePart(source="chart.png")])
        to_openai_content() 输出:
            [
              {"type": "text",      "text": "这张图里有什么？\n图上数字念给我"},
              {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}}
            ]
    """

    instruction: str = Field(default="", description="给模型的指令（会放在最前面）")
    parts: list[ContentPart] = Field(default_factory=list, description="内容片段列表")

    # 【什么】纯文本视图，把图片/文件降级成占位文本
    # 【作用】让不支持多模态的模型也能处理这次输入
    @property
    def text_only(self) -> str:
        """把非文本片段降级成占位文本后的纯文本视图（给不支持视觉的模型）。"""
        pieces = [self.instruction] if self.instruction else []
        pieces.extend(p.to_text() for p in self.parts)
        return "\n".join(x for x in pieces if x)

    # 【什么】本次输入携带的图片张数
    # 【作用】供图片数量上限校验与「是否需要视觉模型」判断使用
    @property
    def image_count(self) -> int:
        return sum(1 for p in self.parts if isinstance(p, ImagePart))

    # 【什么】转成 OpenAI Vision 协议的 content 数组
    # 【作用】把 pydantic 对象翻译成 API 要求的 dict 结构
    def to_openai_content(self) -> list:
        """转成 OpenAI Vision 的 content 数组（文本合并成一条，图片各占一条）。"""
        texts: list[str] = []
        if self.instruction:
            texts.append(self.instruction)
        blocks: list = []
        for part in self.parts:
            if isinstance(part, ImagePart):
                # 图片前先把已攒下的文本冲出去，保证"看图指令"在图片之前
                if texts:
                    blocks.append({"type": "text", "text": "\n".join(texts)})
                    texts = []
                if part.caption:
                    blocks.append({"type": "text", "text": f"[图片{part.caption}]"})
                blocks.append({"type": "image_url", "image_url": {"url": part.data_url}})
            else:
                texts.append(part.to_text())
        if texts:
            blocks.append({"type": "text", "text": "\n".join(texts)})
        return blocks

    # 【什么】整体限额校验器
    # 【作用】卡住一次输入的图片张数，防止多图把上下文撑爆
    @model_validator(mode="after")
    def _check_limits(self) -> "MultiModalInput":
        # ------------------------------------------------------------------
        # 【为什么限制图片张数】视觉模型一次能吃多张图，但张数×每张 token 是相乘的，
        #   5 张高清图轻松吃掉 8k token，直接撞上下文上限。上限由 .env 控制。
        # ------------------------------------------------------------------
        limit = get_settings().vision_max_images
        if self.image_count > limit:
            raise ValueError(f"一次最多携带 {limit} 张图片，当前 {self.image_count} 张")
        return self
