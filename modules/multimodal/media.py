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
import base64  # 导入 base64 模块，用于图片字节与 base64 字符串之间的编解码
import mimetypes  # 导入 mimetypes 模块，用于根据文件扩展名猜测 MIME 类型
from pathlib import Path  # 从 pathlib 导入 Path，提供面向对象的文件路径操作接口
from typing import Annotated, Literal, Union  # 导入类型注解工具：Annotated 用于联合元数据、Literal 用于字面量类型、Union 用于联合类型

from pydantic import BaseModel, Field, model_validator  # 从 pydantic 导入 BaseModel（模型基类）、Field（字段定义）、model_validator（模型级校验器）

from config import get_settings  # 从项目配置模块导入 get_settings，用于读取全局配置（如体积上限、文件字数上限等）

# 【格式白名单】只放行模型侧确认能解的图；不改这一份，别在别处硬编码
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}  # 定义允许的图片 MIME 类型集合，作为全局唯一白名单，防止非法格式流入
# 【文本文件白名单】pdf 走可选提取，其余按纯文本读
ALLOWED_FILE_SUFFIX = {".txt", ".md", ".csv", ".json", ".log", ".pdf"}  # 定义允许的文件后缀集合，pdf 会走特殊文本提取逻辑，其余按纯文本读取


# 【什么】内容片段的公共基类
# 【作用】统一声明 kind 判别字段，并定下「降级为纯文本」的默认行为
# 【为什么】所有内容片段（文本/图片/文件）都需要一个统一入口来标识类型，并提供降级到纯文本的兜底能力
class BasePart(BaseModel):
    """内容片段基类。kind 字段是判别联合的标签，子类各自写死字面量。"""

    kind: str = Field(description="片段类型判别标签：text / image / file")  # 定义片段类型字段，作为判别联合的判别标签，子类会用 Literal 固定具体值

    # 【什么】把本片段降级为纯文本
    # 【作用】给不支持视觉的模型一个兜底视图；基类返回空串，由子类覆写
    # 【为什么】多模态模型并非总是可用，需要统一接口把任何片段转成纯文本，保证文本模型也能消费
    def to_text(self) -> str:
        """降级为纯文本（给不支持视觉的模型用）。子类覆写。"""
        return ""  # 基类默认返回空字符串，表示该片段无文本内容，由具体子类覆写提供真实降级文本


# 【什么】纯文本片段
# 【作用】承载文字内容，无归一化动作，是判别联合里的 text 分支
# 【为什么】文本是最基础的内容类型，不需要额外处理，直接承载原始文字即可
class TextPart(BasePart):
    """文本片段。"""

    kind: Literal["text"] = "text"   # 判别标签，固定为 "text"  # 将 kind 字段固定为字面量 "text"，作为判别联合中识别文本片段的唯一标识
    text: str = Field(description="文本内容")  # 定义文本内容字段，存储用户输入的原始文字

    # 【什么】文本片段的降级视图
    # 【作用】原样返回 text，本身就是文本无需加工
    # 【为什么】文本片段本身就是纯文本，降级视图与原始内容一致，直接返回即可
    def to_text(self) -> str:
        return self.text  # 直接返回存储的文本内容，无需任何转换


# 【什么】图片片段
# 【作用】接收 data URL / 远程 URL / 本地路径三种来源，归一成可直接发送的图片数据
# 【为什么】图片来源多样，需要统一归一化处理成模型可消费的格式，同时做好类型与体积校验
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

    kind: Literal["image"] = "image"  # 将 kind 字段固定为字面量 "image"，作为判别联合中识别图片片段的唯一标识
    source: str = Field(description="data URL / http(s) URL / 本地文件路径")  # 定义图片来源字段，支持三种输入形态：data URL、远程 URL、本地文件路径
    caption: str = Field(default="", description="图片说明（可选，会作为文本一起发给模型）")  # 定义图片说明字段，可选，会在发送时作为文本一并传给模型
    media_type: str = Field(default="", description="归一化后填：image/png 等")  # 定义媒体类型字段，归一化后自动填充具体的 MIME 类型
    data_url: str = Field(default="", description="归一化后可直接发给模型的 URL")  # 定义数据 URL 字段，归一化后填充可直接发给模型的最终地址
    bytes_size: int = Field(default=0, description="原始字节数，用于体积校验")  # 定义字节数字段，记录原始文件大小，用于后续体积上限校验

    # 【什么】实例化后自动执行的归一化校验器
    # 【作用】把 source 的三种形态统一成可直接发送的数据，并卡住类型与体积
    # 【为什么】图片输入来源复杂，必须在实例化时统一处理，避免下游代码重复判断来源类型
    @model_validator(mode="after")  # 使用 pydantic v2 的 model_validator 装饰器，在模型实例化完成后自动执行
    def _normalize(self) -> "ImagePart":
        # ------------------------------------------------------------------
        # 【案例1：远程 URL】不下载。下载要引入网络依赖 + 超时 + SSRF 风险，
        #   交给模型侧自己去取；我们只做协议头校验。
        # ------------------------------------------------------------------
        if self.source.startswith(("http://", "https://")):  # 判断 source 是否以 http:// 或 https:// 开头，识别远程 URL 场景
            self.data_url = self.source  # 远程图片不下载，直接将原始 URL 作为 data_url 透传给模型
            return self  # 远程 URL 场景处理完毕，直接返回实例

        # ------------------------------------------------------------------
        # 【案例2：data URL】拆出 media_type 与 base64 载荷，只校验不重编码。
        #   不这么写的后果：外层再包一层 data:image/png;base64, 前缀，
        #   模型收到双重前缀直接报 400 invalid image。
        # ------------------------------------------------------------------
        if self.source.startswith("data:"):  # 判断 source 是否以 data: 开头，识别 data URL 场景
            header, _, payload = self.source.partition(",")  # 以逗号为分隔符，将 data URL 拆分为头部（含类型信息）和 base64 载荷
            if ";base64" not in header or not payload:  # 校验头部必须包含 ;base64 标记且载荷非空，否则格式非法
                raise ValueError("data URL 格式非法，应为 data:<type>;base64,<payload>")  # 抛出 ValueError 提示正确的 data URL 格式
            self.media_type = header[len("data:"):].split(";")[0]  # 从头部提取 MIME 类型：去掉 "data:" 前缀后按分号切分取第一段
            self.data_url = self.source  # data URL 已经是模型可直接消费的格式，直接透传
            self.bytes_size = self._decoded_size(payload)  # 调用静态方法计算 base64 解码后的真实字节数
        else:
            # ------------------------------------------------------------------
            # 【案例3：本地路径】读盘 → 猜 MIME → 转 base64 → 组装 data URL。
            #   不这么写的后果：模型服务在内网/Docker 里根本访问不到你的本地路径。
            # ------------------------------------------------------------------
            path = Path(self.source)  # 将字符串路径转换为 Path 对象，便于后续文件操作
            if not path.exists():  # 检查文件是否存在，不存在则无法读取
                raise ValueError(f"图片文件不存在: {self.source}")  # 抛出 ValueError 提示文件路径无效
            raw = path.read_bytes()  # 以二进制模式读取文件全部字节内容
            self.bytes_size = len(raw)  # 记录原始文件字节数，用于后续体积校验
            guess = mimetypes.guess_type(path.name)[0] or ""  # 根据文件名猜测 MIME 类型，猜不到则置空字符串
            self.media_type = guess if guess in ALLOWED_IMAGE_TYPES else ""  # 只有猜测类型在白名单内才采纳，否则置空（后续会触发类型校验错误）
            self.data_url = (
                f"data:{self.media_type};base64,{base64.b64encode(raw).decode('ascii')}"  # 将二进制数据编码为 base64 并组装成标准 data URL
            )

        if self.media_type and self.media_type not in ALLOWED_IMAGE_TYPES:  # 如果已识别出媒体类型且不在白名单内
            raise ValueError(
                f"不支持的图片类型 {self.media_type}，允许: {sorted(ALLOWED_IMAGE_TYPES)}"  # 抛出 ValueError 提示不支持的类型及允许列表
            )

        limit_mb = get_settings().vision_max_image_mb  # 从全局配置读取图片体积上限（单位：MB）
        if self.bytes_size and self.bytes_size > limit_mb * 1024 * 1024:  # 如果字节数非零且超过上限（转换为字节比较）
            raise ValueError(
                f"图片 {self.bytes_size / 1048576:.2f}MB 超过上限 {limit_mb}MB"  # 抛出 ValueError 提示当前大小与上限
            )
        return self  # 所有校验与归一化完成，返回实例

    # 【什么】按 base64 载荷反算图片的原始字节数
    # 【作用】给体积上限校验提供数据；解码失败返回 0（0 表示跳过体积校验）
    # 【为什么】data URL 中的 base64 字符串需要解码后才能知道真实大小，但解码失败不应阻断流程
    @staticmethod  # 声明为静态方法，不依赖实例状态
    def _decoded_size(payload: str) -> int:
        """算 base64 载荷解码后的真实字节数（校验失败返回 0，不抛异常）。"""
        try:
            return len(base64.b64decode(payload, validate=False))  # 尝试解码 base64 并返回字节长度；validate=False 允许非标准 base64
        except Exception:
            return 0  # 解码失败（如载荷损坏）时返回 0，表示跳过体积校验，不阻断主流程

    # 【什么】图片片段降级成占位文本
    # 【作用】带 caption 时输出「[图片：说明]」，无 caption 时输出「[图片]」
    # 【为什么】文本模型无法理解图片，需要一个人类可读的占位符告知"这里有一张图片"
    def to_text(self) -> str:
        return f"[图片{self.caption and '：' + self.caption}]"  # 使用 f-string 格式化：有 caption 时输出 [图片：说明]，无 caption 时输出 [图片]


# 【什么】文件片段
# 【作用】在实例化时就把文件读成文本：文本文件直接读，pdf 走可选抽取
# 【为什么】文件内容需要提前加载并转成文本，避免下游处理时反复做 IO 和格式判断
class FilePart(BasePart):
    """文件片段：文本文件直接读成字符串；pdf 尝试用 pypdf 抽文本，抽不到降级占位。"""

    kind: Literal["file"] = "file"  # 将 kind 字段固定为字面量 "file"，作为判别联合中识别文件片段的唯一标识
    path: str = Field(description="本地文件路径")  # 定义文件路径字段，存储本地文件的字符串路径
    name: str = Field(default="", description="展示用文件名")  # 定义文件名字段，用于展示，默认从路径中提取
    content: str = Field(default="", description="抽取出的文本（截断到 max_file_chars）")  # 定义内容字段，存储从文件中抽取的文本，会按配置截断
    bytes_size: int = Field(default=0)  # 定义字节数字段，记录文件原始大小

    # 【什么】实例化后立即读盘并抽取文本
    # 【作用】保证 downstream 拿到的 FilePart 一定带 content，无需再判空或处理 IO
    # 【为什么】文件读取与文本抽取是耗时操作，在实例化时一次性完成，避免下游重复劳动
    @model_validator(mode="after")  # 使用 pydantic v2 的 model_validator 装饰器，在模型实例化完成后自动执行
    def _load(self) -> "FilePart":
        path = Path(self.path)  # 将字符串路径转换为 Path 对象
        if not path.exists():  # 检查文件是否存在
            raise ValueError(f"文件不存在: {self.path}")  # 抛出 ValueError 提示文件路径无效
        if path.suffix.lower() not in ALLOWED_FILE_SUFFIX:  # 检查文件后缀（转小写后）是否在白名单内
            raise ValueError(f"不支持的文件类型 {path.suffix}，允许: {sorted(ALLOWED_FILE_SUFFIX)}")  # 抛出 ValueError 提示不支持的类型
        self.name = self.name or path.name  # 如果未指定展示名，则使用路径中的文件名
        self.bytes_size = path.stat().st_size  # 获取文件状态并记录字节数
        self.content = self._extract(path)  # 调用静态方法按后缀抽取文件正文
        return self  # 加载完成，返回实例

    # 【什么】按后缀抽取文件正文
    # 【作用】pdf 走 pypdf 抽文本，其余按 utf-8 读文本；两种情况都截断到 limit 字
    # 【为什么】不同文件格式需要不同的解析策略，统一封装在此方法中，对上游透明
    @staticmethod  # 声明为静态方法，不依赖实例状态
    def _extract(path: Path) -> str:
        limit = get_settings().vision_max_file_chars   # 文本抽取上限（字），来自配置 vision_max_file_chars  # 从全局配置读取文本抽取字数上限
        if path.suffix.lower() == ".pdf":  # 判断是否为 PDF 文件（后缀转小写后比较）
            try:
                from pypdf import PdfReader  # 延迟导入 pypdf 库的 PdfReader，避免未安装时影响其他功能
                text = "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)  # 遍历 PDF 每一页提取文本，空页返回空串，用换行拼接
                return text[:limit]  # 按配置上限截断后返回
            except Exception:
                # pdf 抽取失败不阻断主流程：留占位，让 LLM 知道"有这个文件但没读到"
                return f"（PDF 文本抽取失败：{path.name}）"  # 捕获所有异常（包括 pypdf 未安装、PDF 损坏等），返回占位文本
        try:
            return path.read_text(encoding="utf-8", errors="ignore")[:limit]  # 以 UTF-8 编码读取文本文件，忽略解码错误，按上限截断
        except Exception:
            return f"（文件读取失败：{path.name}）"  # 捕获所有异常（如权限不足、文件损坏等），返回占位文本

    # 【什么】文件片段降级成「文件名 + 正文」文本
    # 【作用】不支持文件的模型也能看到文件名与已抽取的内容
    # 【为什么】文本模型无法直接接收文件，需要把文件信息转成纯文本形式传递
    def to_text(self) -> str:
        return f"[文件 {self.name}]\n{self.content}"  # 使用 f-string 格式化：首行显示文件名，次行显示抽取的正文内容


# 【判别联合】靠 kind 字段选类型，避免字段相似导致的错误匹配
ContentPart = Annotated[
    Union[TextPart, ImagePart, FilePart],  # 定义三种片段的联合类型
    Field(discriminator="kind"),  # 使用 pydantic Field 的 discriminator 参数指定以 kind 字段作为判别依据
]


# 【什么】一次完整的多模态输入
# 【作用】把「指令 + 若干内容片段」打包，并负责整体限额校验与协议转换
# 【为什么】多模态请求通常包含一条指令和多个内容片段，需要一个顶层容器统一管理并做整体校验
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

    instruction: str = Field(default="", description="给模型的指令（会放在最前面）")  # 定义指令字段，存储用户给模型的文字指令，默认空串
    parts: list[ContentPart] = Field(default_factory=list, description="内容片段列表")  # 定义片段列表字段，使用判别联合类型，默认空列表

    # 【什么】纯文本视图，把图片/文件降级成占位文本
    # 【作用】让不支持多模态的模型也能处理这次输入
    # 【为什么】当视觉模型不可用时，需要一种方式把多模态输入转成纯文本，让文本模型也能处理
    @property  # 声明为属性，可像访问字段一样调用
    def text_only(self) -> str:
        """把非文本片段降级成占位文本后的纯文本视图（给不支持视觉的模型）。"""
        pieces = [self.instruction] if self.instruction else []  # 如果指令非空，则作为第一段文本；否则从空列表开始
        pieces.extend(p.to_text() for p in self.parts)  # 遍历所有片段，调用各自的 to_text() 降级方法，追加到文本列表
        return "\n".join(x for x in pieces if x)  # 过滤空串后用换行符拼接，返回最终纯文本

    # 【什么】本次输入携带的图片张数
    # 【作用】供图片数量上限校验与「是否需要视觉模型」判断使用
    # 【为什么】图片张数直接影响 token 消耗与模型选择，需要快速统计
    @property  # 声明为属性，可像访问字段一样调用
    def image_count(self) -> int:
        return sum(1 for p in self.parts if isinstance(p, ImagePart))  # 生成器表达式遍历片段，统计 ImagePart 实例的数量

    # 【什么】转成 OpenAI Vision 协议的 content 数组
    # 【作用】把 pydantic 对象翻译成 API 要求的 dict 结构
    # 【为什么】OpenAI Vision API 有固定的消息格式要求，需要将内部模型转换为该格式
    def to_openai_content(self) -> list:
        """转成 OpenAI Vision 的 content 数组（文本合并成一条，图片各占一条）。"""
        texts: list[str] = []  # 初始化文本暂存列表，用于累积连续的文本片段
        if self.instruction:  # 如果指令非空
            texts.append(self.instruction)  # 将指令作为第一段文本加入暂存列表
        blocks: list = []  # 初始化结果块列表，存储最终按 OpenAI 协议格式化的内容块
        for part in self.parts:  # 遍历所有内容片段
            if isinstance(part, ImagePart):  # 如果当前片段是图片
                # 图片前先把已攒下的文本冲出去，保证"看图指令"在图片之前
                if texts:  # 如果暂存列表中有累积的文本
                    blocks.append({"type": "text", "text": "\n".join(texts)})  # 将累积文本合并为一条 text 类型块加入结果
                    texts = []  # 清空暂存列表，准备累积下一段文本
                if part.caption:  # 如果图片有说明文字
                    blocks.append({"type": "text", "text": f"[图片{part.caption}]"})  # 将图片说明作为一条 text 类型块加入结果
                blocks.append({"type": "image_url", "image_url": {"url": part.data_url}})  # 将图片数据作为 image_url 类型块加入结果
            else:
                texts.append(part.to_text())  # 非图片片段（文本/文件）调用 to_text() 降级后追加到暂存列表
        if texts:  # 循环结束后如果还有未冲出的文本
            blocks.append({"type": "text", "text": "\n".join(texts)})  # 将剩余文本合并为一条 text 类型块加入结果
        return blocks  # 返回按 OpenAI Vision 协议格式化的内容块列表

    # 【什么】整体限额校验器
    # 【作用】卡住一次输入的图片张数，防止多图把上下文撑爆
    # 【为什么】图片张数与 token 消耗成正比，必须限制单次请求的图片数量，避免超出模型上下文上限
    @model_validator(mode="after")  # 使用 pydantic v2 的 model_validator 装饰器，在模型实例化完成后自动执行
    def _check_limits(self) -> "MultiModalInput":
        # ------------------------------------------------------------------
        # 【为什么限制图片张数】视觉模型一次能吃多张图，但张数×每张 token 是相乘的，
        #   5 张高清图轻松吃掉 8k token，直接撞上下文上限。上限由 .env 控制。
        # ------------------------------------------------------------------
        limit = get_settings().vision_max_images  # 从全局配置读取单次请求允许的最大图片张数
        if self.image_count > limit:  # 如果当前输入的图片张数超过上限
            raise ValueError(f"一次最多携带 {limit} 张图片，当前 {self.image_count} 张")  # 抛出 ValueError 提示超限
        return self  # 校验通过，返回实例
