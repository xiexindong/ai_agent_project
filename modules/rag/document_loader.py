"""
文档加载模块
============
支持 TXT / MD / PDF 多格式文档加载，提取纯文本。
"""
from pathlib import Path                                     # 导入 Path：面向对象的文件路径处理，自动兼容各操作系统的分隔符
from typing import List                                      # 导入 List 泛型：给批量加载的返回值做类型标注
from dataclasses import dataclass                            # 导入 dataclass 装饰器：自动生成 __init__ 等样板代码
from modules.common.logger import get_logger                 # 导入统一日志工厂：保证全项目日志格式一致、便于集中收集

_logger = get_logger("doc_loader")                          # 本模块日志器，命名 doc_loader，便于在日志里定位来源模块


# 【什么】纯数据容器：一份文档 = 正文 content + 元数据 metadata
# 【作用】在加载器 / 切分器 / 向量库之间传递文档，避免到处用裸 dict / tuple 传参
# 【为什么】用 @dataclass 自动生成构造方法，字段声明一目了然，调用方必须关键字传参不易错位
@dataclass                                                  # dataclass 装饰器：按下面声明的字段自动生成 __init__/__repr__
class Document:                                             # 文档数据结构类定义
    """文档数据结构：内容 + 元数据。"""
    content: str                                            # 文档提取后的纯文本正文
    metadata: dict                                          # 元数据字典：source / file_type / file_name / doc_id 等


# 【什么】多格式文档加载器
# 【作用】RAG 入库链路的第 1 步：把磁盘上的 TXT / MD / PDF 统一读成内存里的 Document
# 【为什么】上层（RAGChain）不该关心文件格式差异；格式判断收敛在这一个类里，新增格式只改这里
class DocumentLoader:                                       # 加载器类定义：全部是类方法/静态方法，无需实例化
    """多格式文档加载器。"""

    SUPPORTED_TYPES = {".txt", ".md", ".pdf"}               # 支持的扩展名集合：用 set 让 in 判断 O(1)，统一小写便于后缀比较

    # 【什么】单文件加载入口
    # 【作用】给一个文件路径，返回解析好的 Document（自动识别格式）
    # 【为什么】先校验存在性与格式白名单再分支解析：坏输入在最早处报错，错误信息指得准、不污染下游
    @classmethod                                            # 类方法：不依赖实例状态，直接 DocumentLoader.load(...) 调用
    def load(cls, file_path: str) -> Document:              # 加载入口：传文件路径字符串，返回 Document
        """加载单个文档，自动识别格式。"""
        path = Path(file_path)                              # 字符串路径包装成 Path 对象，后续取后缀/文件名更方便
        if not path.exists():                               # 先查文件是否存在
            raise FileNotFoundError(f"文档不存在: {file_path}")  # 不存在直接抛错：带上原始路径，方便定位是哪个文件丢了

        suffix = path.suffix.lower()                        # 取扩展名并转小写：.PDF 与 .pdf 视为同一种格式
        if suffix not in cls.SUPPORTED_TYPES:               # 白名单校验：不支持的格式拒绝处理
            raise ValueError(f"不支持的文档格式: {suffix}，仅支持 {cls.SUPPORTED_TYPES}")  # 报错里带上支持列表，调用方一看就知道怎么办

        # 格式分派：PDF 走专用解析器；MD / TXT 本质都是纯文本，直接按 utf-8 读
        if suffix == ".pdf":                                # PDF 分支
            content = cls._load_pdf(path)                   # 调 pypdf 逐页提取文字
        elif suffix == ".md":                               # Markdown 分支
            content = cls._load_text(path)                  # md 是纯文本，直接读
        else:                                               # 剩下的只有 .txt（白名单已过滤过其他可能）
            content = cls._load_text(path)                  # txt 同样按纯文本读

        doc = Document(                                     # 组装文档对象
            content=content,                                # 正文：上面解析出的纯文本
            metadata={                                      # 元数据：下游切分/入库/溯源都要用
                "source": str(path),                        # 完整路径：检索结果的"来源"展示靠它
                "file_type": suffix,                        # 文件类型：统计与排查用
                "file_name": path.name,                     # 文件名：上下文拼接时展示给用户看
                "doc_id": path.stem,                        # 文档 id：取去后缀的文件名，下线文档时按它删除
            },
        )
        _logger.info(f"文档加载完成: {path.name} | 字符数={len(content)}")  # 记录加载结果：字符数为 0 是重要异常信号
        return doc                                          # 返回组装好的文档对象

    # 【什么】目录批量加载入口
    # 【作用】递归扫描目录，把所有支持格式的文件全部加载成 Document 列表
    # 【为什么】单个文件失败不拖垮整批（try/except 包住单个文件）：100 个文件里 1 个损坏不该让其余 99 个白跑
    @classmethod                                            # 类方法：同样无需实例化
    def load_batch(cls, dir_path: str) -> List[Document]:   # 批量加载入口：传目录路径，返回文档列表
        """批量加载目录下所有支持格式的文档。"""
        directory = Path(dir_path)                          # 目录路径包装成 Path 对象
        if not directory.is_dir():                          # 校验传入的确实是目录
            raise NotADirectoryError(f"路径不是目录: {dir_path}")  # 不是目录直接抛错（比如把文件路径当目录传了）

        docs = []                                           # 累积成功加载的文档
        for path in directory.rglob("*"):                   # rglob 递归遍历所有子目录
            if path.is_file() and path.suffix.lower() in cls.SUPPORTED_TYPES:  # 只处理"是文件"且"格式受支持"的条目
                try:                                        # 单个文件加载用 try 包住：坏文件不中断整批
                    docs.append(cls.load(str(path)))        # 复用单文件加载逻辑，成功后入列表
                except Exception as e:                      # 捕获该文件的一切加载异常（损坏/编码错误等）
                    _logger.error(f"加载失败 {path.name}: {e}")  # 记错误日志后跳过：事后可按日志补录
        _logger.info(f"批量加载完成: 共 {len(docs)} 个文档")  # 汇总日志：成功数与目录文件数对不上 = 有文件加载失败
        return docs                                         # 返回成功加载的文档列表（可能少于目录内文件总数）

    # 【什么】纯文本读取
    # 【作用】把 TXT / MD 文件读成字符串
    # 【为什么】独立成静态方法：两种格式共用同一段读取逻辑，且显式指定 utf-8 避免 Windows 默认 gbk 读中文乱码
    @staticmethod                                           # 静态方法：纯函数，不碰类状态
    def _load_text(path: Path) -> str:                      # 文本读取：Path -> 字符串
        """加载 TXT / MD 文本文件。"""
        return path.read_text(encoding="utf-8")             # 统一按 utf-8 解码读取

    # 【什么】PDF 文本提取
    # 【作用】把 PDF 每一页的文字抽出来拼成一个字符串
    # 【为什么】pypdf 是可选依赖：只有真用到 PDF 才在函数内 import，没装它的机器上 TXT/MD 功能照常可用
    @staticmethod                                           # 静态方法：不依赖类/实例状态
    def _load_pdf(path: Path) -> str:                       # PDF 读取：Path -> 字符串
        """加载 PDF 文档（基于 pypdf）。"""
        try:                                                # 延迟导入放在 try 里
            from pypdf import PdfReader                     # 函数内 import：用到 PDF 功能才加载这个库
        except ImportError:                                 # 没装 pypdf
            raise ImportError("请安装 pypdf: pip install pypdf")  # 给出明确的安装指引再抛出

        reader = PdfReader(str(path))                       # 打开 PDF 文件（PdfReader 接收字符串路径）
        pages = [page.extract_text() or "" for page in reader.pages]  # 逐页提取文字；扫描件某页可能返回 None，用 or "" 兜底防空
        return "\n".join(pages).strip()                     # 页与页之间用换行拼接，去掉首尾空白后返回
