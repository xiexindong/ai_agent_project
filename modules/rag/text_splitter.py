"""
文本切分模块
============
递归字符分割 + 重叠分片，保障上下文完整性。
"""
from typing import List                                      # 导入 List 泛型：标注返回值类型
from dataclasses import dataclass                            # 导入 dataclass 装饰器：快速定义数据类
from modules.rag.document_loader import Document             # 引入 Document：split_document 的入参类型
from modules.common.logger import get_logger                 # 导入统一日志工厂

_logger = get_logger("splitter")                            # 本模块日志器，命名 splitter，便于日志定位来源


# 【什么】文本分片数据容器
# 【作用】承载"一小段可入库的文本"及其在原文档中的位置信息
# 【为什么】带上 chunk_index / total_chunks 后，检索命中时能溯源"这是哪篇文档的第几片"，
#   还能按序号把相邻片拼回上下文
@dataclass                                                  # dataclass 装饰器：自动生成 __init__ 等样板代码
class TextChunk:                                            # 分片类定义
    """文本分片：内容 + 元数据 + 序号。"""
    content: str                                            # 分片正文
    metadata: dict                                          # 元数据：继承自文档，另加 chunk_id
    chunk_index: int                                        # 本片在文档中的序号（从 0 起）
    total_chunks: int                                       # 文档总片数：便于判断"第几片/共几片"


# 【什么】递归字符切分器
# 【作用】把长文档切成不超过 chunk_size 的小段，相邻段保留 chunk_overlap 重叠
# 【为什么】① embedding 模型有输入长度上限，且片段越小向量语义越聚焦、检索越精准；
#   ② 重叠是为了不让"正好被切断的句子"丢失上下文——答案跨在切缝上时两片各有半截
class TextSplitter:                                         # 切分器类定义
    """递归字符分割器，支持重叠分片。"""

    DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]  # 分隔符优先级表：先段落、再行、再中英文句号/空格，最后空串硬切——尽量在语义边界下刀

    # 【什么】构造器：固定切分参数
    # 【作用】保存本实例的 chunk_size / chunk_overlap / 分隔符表
    # 【为什么】不同场景对粒度要求不同（问答要细、摘要可粗），做成构造参数而不是全局常量
    def __init__(
        self,
        chunk_size: int = 500,                              # 每片最大字符数：500 是"语义完整"与"检索精准"的折中
        chunk_overlap: int = 80,                            # 相邻片重叠字符数：保证跨片句子不断裂
        separators: list | None = None,                     # 自定义分隔符表；传 None 用默认表
    ):
        self.chunk_size = chunk_size                        # 保存片大小上限
        self.chunk_overlap = chunk_overlap                  # 保存重叠长度
        self.separators = separators or self.DEFAULT_SEPARATORS  # 没传自定义表就用默认分隔符表

    # 【什么】纯文本切分入口
    # 【作用】把一段字符串切成若干不超过 chunk_size 的片段
    # 【为什么】拆成"先递归切、再补重叠"两步：切分与重叠解耦，overlap=0 时就是纯切分，逻辑各自可测
    def split_text(self, text: str) -> List[str]:           # 入口：传整段文本，返回片段字符串列表
        """递归切分文本，保证分片不超过 chunk_size，相邻分片有 overlap 重叠。"""
        if len(text) <= self.chunk_size:                    # 文本本身就不超长
            return [text] if text.strip() else []           # 非空白直接单片返回；全空白返回空列表（不入库）

        chunks = []                                         # 收集切出的片段
        self._recursive_split(text, chunks)                 # 递归切分，结果追加进 chunks
        if self.chunk_overlap > 0 and len(chunks) > 1:      # 开了重叠且切出了多片
            chunks = self._add_overlap(chunks)              # 给相邻片补上前一片的尾巴
        return chunks                                       # 返回最终片段列表

    # 【什么】递归切分核心
    # 【作用】按 separators[depth] 切，切不动的超长段换下一级更细的分隔符递归
    # 【为什么】"先大段后小句"的贪心合并：让每片尽量装满 chunk_size、又不断在句子中间——语义边界优先
    def _recursive_split(self, text: str, chunks: list, depth: int = 0):  # depth = 当前用到第几级分隔符
        """递归按分隔符切分。"""
        if len(text) <= self.chunk_size:                    # 递归出口 1：这段已经够短
            if text.strip():                                # 且不是全空白
                chunks.append(text.strip())                 # 收进结果（顺手去掉首尾空白）
            return                                          # 本段处理完毕
        if depth >= len(self.separators):                   # 递归出口 2：分隔符用完了还超长
            for i in range(0, len(text), self.chunk_size):  # 只能按 chunk_size 机械硬切
                piece = text[i:i + self.chunk_size].strip() # 取一段并去空白
                if piece:                                   # 跳过纯空白段
                    chunks.append(piece)                    # 收进结果
            return                                          # 硬切完毕

        sep = self.separators[depth]                        # 取当前级分隔符
        if sep == "":                                       # 空分隔符 = 最后一级，语义同硬切
            for i in range(0, len(text), self.chunk_size):  # 按 chunk_size 逐段切
                piece = text[i:i + self.chunk_size].strip() # 取一段并去空白
                if piece:                                   # 跳过纯空白段
                    chunks.append(piece)                    # 收进结果
            return                                          # 处理完毕

        # 贪心合并设计：把 split 出来的小段逐个往当前片上拼，拼到刚好不超长为止——
        # 让每片信息量最大化（片数少 = 入库向量少 = 检索快），同时切口都落在分隔符上
        parts = text.split(sep)                             # 按当前分隔符切开
        current = ""                                        # 正在累积的当前片
        for part in parts:                                  # 逐个片段尝试并入当前片
            candidate = current + sep + part if current else part  # 试着拼到当前片后面（拼回被 split 吃掉的分隔符，保持原文）
            if len(candidate) <= self.chunk_size:           # 拼上之后仍不超长
                current = candidate                         # 那就拼上：让每片尽量装满
            else:                                           # 拼上就超长了
                if current.strip():                         # 当前片已有内容
                    chunks.append(current.strip())          # 先收进结果
                if len(part) > self.chunk_size:             # 单个片段本身就超长（比如一个超长自然段）
                    self._recursive_split(part, chunks, depth + 1)  # 用下一级更细的分隔符递归切它
                    current = ""                            # 当前片清空，从该片段之后重新累积
                else:                                       # 片段本身不超长
                    current = part                          # 它成为新一片的开头
        if current.strip():                                 # 循环结束后当前片可能还有残余
            chunks.append(current.strip())                  # 收进结果，别丢了最后一片

    # 【什么】相邻片重叠补丁
    # 【作用】给第 2 片起每片前面拼上前一片的末尾 chunk_overlap 个字符
    # 【为什么】防止"答案正好跨在切缝上"：被切断的句子在两片里各有一半，检索任一片都能带上完整语义
    def _add_overlap(self, chunks: List[str]) -> List[str]: # 传入切好的片段，返回加过重叠的片段
        """给相邻分片添加重叠，保留上下文连贯性。"""
        result = [chunks[0]]                                # 第 1 片原样保留（它前面没有东西可叠）
        for i in range(1, len(chunks)):                     # 从第 2 片开始逐片处理
            prev_tail = chunks[i - 1][-self.chunk_overlap:] # 取前一片的末尾 overlap 个字符
            result.append(prev_tail + chunks[i])            # 拼到本片前面
        return result                                       # 返回加叠后的列表

    # 【什么】文档级切分入口
    # 【作用】把 Document 切成带完整元数据的 TextChunk 列表
    # 【为什么】split_text 只管切字符串；这里负责把文档元数据透传到每一片并生成 chunk_id——
    #   入库、检索命中后的溯源展示都靠这些元数据
    def split_document(self, doc: Document) -> List[TextChunk]:  # 传入文档对象，返回分片对象列表
        """切分文档，返回带元数据的分片列表。"""
        pieces = self.split_text(doc.content)               # 先按纯文本逻辑切出片段
        total = len(pieces)                                 # 总片数：写进每一片的 total_chunks
        chunks = [                                          # 列表推导：逐片包装成 TextChunk
            TextChunk(
                content=p,                                  # 分片正文
                metadata={**doc.metadata, "chunk_id": f"{doc.metadata.get('doc_id', 'doc')}_{i}"},  # 透传文档元数据并生成 chunk_id：文档id_序号，全局唯一
                chunk_index=i,                              # 本片序号
                total_chunks=total,                         # 总片数
            )
            for i, p in enumerate(pieces)                   # 带序号遍历所有片段
        ]
        _logger.info(                                       # 切分结果日志
            f"文档切分完成: {doc.metadata.get('file_name', 'unknown')} | "  # 哪个文档
            f"分片数={total} | chunk_size={self.chunk_size} | overlap={self.chunk_overlap}"  # 切出几片、用的什么参数
        )
        return chunks                                       # 返回分片列表
