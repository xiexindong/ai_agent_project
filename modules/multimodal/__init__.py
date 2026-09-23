"""多模态子包：内容模型（media）+ 视觉调用（vision）。"""

# 【什么】从 media 导入白名单常量与全部内容片段类型
# 【作用】外部构造多模态输入只用这几个类，不必记具体模块路径
from modules.multimodal.media import (
    ALLOWED_IMAGE_TYPES,
    ALLOWED_FILE_SUFFIX,
    BasePart,
    TextPart,
    ImagePart,
    FilePart,
    ContentPart,
    MultiModalInput,
)

# 【什么】对外可见的名字清单
# 【作用】from modules.multimodal import * 时只导出这些，不泄露内部符号
__all__ = [
    "ALLOWED_IMAGE_TYPES",
    "ALLOWED_FILE_SUFFIX",
    "BasePart",
    "TextPart",
    "ImagePart",
    "FilePart",
    "ContentPart",
    "MultiModalInput",
]
