"""
llama-index 适配层（可选后端，默认不启用）
==========================================
【定位】
    这一层不替代自研 RAGChain，而是提供"第二种检索实现"。
    自研链路（Milvus + BGE 精排 + 权限 expr）是默认路径；
    llama-index 只在显式配置 RAG_BACKEND=llamaindex 时接管。

【为什么用适配层而不是直接用 llama-index 重写 RAG】
    1. 上层 ReActAgent / API / Harness 全部依赖 RAGChain 的三个方法：
       ingest_document / retrieve / build_context。换成别的库就改一片调用点，
       是典型的"框架渗透进业务代码"，将来再换回来自研就又要改一遍。
    2. 自研链路里有两样东西框架不自带：权限过滤（kb_code in [...]）和
       BGE 精排。适配层必须把这两样补回去，否则换库 = 越权 + 掉精度。
    3. 它是可选依赖：没装库时 import 会失败，适配层要做成"能用就用，
       不能用就明确报错让上层降级"，而不是整个服务起不来。

【依赖警告】
    llama-index 0.10 前后 API 大改（llama_index → llama_index.core），
    本文件按新 API（import from llama_index.core）写，并对旧路径做了兜底。
    装的是 <0.10 的老版本时，走 _import_legacy() 兜底分支。
"""
import os                                                   # 标准库 os：检测文件存在、创建目录
from typing import Optional                                 # 类型标注：Optional 可选参数

from modules.common.logger import get_logger                # 统一日志工厂
from modules.common.exceptions import RAGException          # RAG 业务异常：适配层无法翻译的权限表达式必须拒绝
from config import get_settings                             # 全局配置读取

_logger = get_logger("llamaindex_adapter")                  # 本模块日志器，命名 llamaindex_adapter，便于日志定位来源


# 【什么】运行时可用性探测
# 【作用】判断 llama-index 新 API 是否已安装
# 【为什么】工厂在 build_rag() 里调用它来决定是否走适配层；import 失败时不抛错，仅返回 False——
#   探活接口不该把可选依赖缺失变成服务启动异常
def is_available() -> bool:                                 # 返回布尔值：库装了就 True，没装 False
    """检测 llama-index 是否可用（供工厂与 /agent/backends 探活用）。"""
    try:
        import llama_index.core  # noqa: F401              # 只 import 不实际使用，验证新 API 路径是否存在
        return True                                         # import 成功 = 新 API 可用
    except Exception:                                       # 任何失败（ImportError / AttributeError / 子依赖缺）
        return False                                        # 静默返回 False，允许上层降级到自研链路


# ------------------------------------------------------------------
# 【权限表达式翻译】Milvus expr → llama-index 过滤条件
#   自研链路用 Milvus 布尔表达式做知识库隔离（如 kb_code in ["kb_a","kb_b"]），
#   llama-index 用的是 MetadataFilters。两套语法必须翻译，
#   翻译不了就必须拒绝查询（fail closed）。
#
# 【为什么不翻译失败就退回全量检索】
#   过滤失效 = 越权文档进入模型上下文 = 泄密。宁可这次查不出东西，
#   也不能让不该看的内容被召回。空结果用户会重试，泄密无法撤回。
#
# 【输入输出对照】
#   输入：'kb_code in ["kb_a","kb_b"]'
#   输出：[{"key":"kb_code","op":"in","value":["kb_a","kb_b"]}]
#   输入：'level == "public" and kb_code in ["kb_a"]'
#   输出：[{"key":"level","op":"==","value":"public"},
#          {"key":"kb_code","op":"in","value":["kb_a"]}]
#   输入：'score > 0.5' → 抛 RAGException（不支持的比较，拒绝放行）
# ------------------------------------------------------------------
# 【什么】Milvus 表达式解析器
# 【作用】把 Milvus 风格的过滤字符串翻译成中立的字典列表，再由 LlamaIndexRAG._filters 转成 llama-index 对象
# 【为什么】① 不依赖 llama-index 的可先写单元测试；② 翻译失败必须抛 RAGException（fail closed）
#   因为"翻译不出" ≠ "不过滤"，翻译成"不过滤"等于越权
def parse_expr(expr: Optional[str]) -> list:               # 传字符串表达式，返回字典列表
    """把 Milvus 风格的过滤表达式解析成中立结构（不依赖 llama-index，可单测）。"""
    if not expr or not expr.strip():                        # 空/空白表达式 = 不过滤
        return []                                           # 返回空列表：_filters 拿到空列表会返回 None（不设过滤）
    parsed = []                                             # 收集解析出的条件字典
    for clause in expr.split(" and "):                      # 目前只支持 and 连接多条件；不支持 or
        clause = clause.strip()                             # 去掉首尾空白
        if not clause:                                      # 跳过空段（比如连续多个 and 产生的空白）
            continue                                        # 继续下一条
        if " in [" in clause:                               # 分支 1：in 操作（kb_code in [...]）
            key, raw = clause.split(" in [", 1)             # 按 " in [" 劈成键和值
            if not raw.endswith("]"):                       # 右括号缺失
                raise RAGException(f"过滤表达式缺少右括号，拒绝放行（防越权）: {clause}")  #  fail closed：不猜语义，直接拒绝
            values = [v.strip().strip('"').strip("'") for v in raw[:-1].split(",")]  # 去掉末尾 ] 后按逗号拆，每个值去空白去引号
            parsed.append({"key": key.strip(), "op": "in", "value": [v for v in values if v]})  # 过滤空值后入列
        elif "==" in clause:                                # 分支 2：等于（如 level == "public"）
            key, val = clause.split("==", 1)                # 按 == 劈成键和值
            parsed.append({"key": key.strip(), "op": "==", "value": val.strip().strip('"').strip("'")})  # 去空白去引号后入列
        elif "!=" in clause:                                # 分支 3：不等于
            key, val = clause.split("!=", 1)                # 按 != 劈成键和值
            parsed.append({"key": key.strip(), "op": "!=", "value": val.strip().strip('"').strip("'")})  # 去空白去引号后入列
        else:                                               # 分支 4：其他操作符（> / < / like / or 等）
            # 不支持 or / like / 比较符：不猜，直接拒绝
            raise RAGException(f"过滤表达式无法翻译，拒绝放行（防越权）: {clause}")  # fail closed：宁可漏结果也不泄密
    return parsed                                           # 返回中立结构列表


# 【什么】llama-index 版 RAG 实现
# 【作用】提供与自研 RAGChain 完全相同的三个方法签名，上层调用方无感知替换后端
# 【为什么】保持接口一致性是适配层的核心使命：换了后端，上游 ReActAgent / API 一行不改
class LlamaIndexRAG:
    """llama-index 版检索实现：接口与自研 RAGChain 对齐，可无缝替换。"""

    # 【什么】构造器：初始化 llama-index 环境
    # 【作用】确定落盘目录、获取检索参数、k 值，并延迟导入 llama-index 类
    # 【为什么】persist_dir 用默认值 fallback：没配 llamaindex_persist_dir 时取 "storage/llamaindex"，
    #   保证首次运行不因为缺配置抛错；top_k / rerank_top_k 从全局配置取
    def __init__(self, persist_dir: str = "", top_k: int = 0, rerank_top_k: int = 0):
        s = get_settings()                                  # 取全局配置对象
        self.persist_dir = persist_dir or getattr(s, "llamaindex_persist_dir", "storage/llamaindex")  # persist_dir 优先用传入值，空则读配置，再空取默认目录
        self._top_k = top_k or s.rag_top_k                  # 召回条数：参数 > 配置 > 缺省
        self._rerank_top_k = rerank_top_k or s.rerank_top_k  # 精排条数
        self._li = self._import_li()                        # 延迟导入 llama-index 核心类，存成字典供后续用
        self._index = None                                  # 索引句柄：懒加载，首次调用 _ensure_index 时才创建或加载

    # 【什么】延迟导入
    # 【作用】把 llama-index 核心类包进字典，供构造后各方法调用
    # 【为什么】函数内 import：可选依赖在构造函数里才触碰 import；模块级 import 会让没装 llama-index 的进程在 import 本文件时就抛错，违背"可选"设计
    @staticmethod                                           # 静态方法：纯工具函数，不依赖实例状态
    def _import_li() -> dict:                               # 返回字典：键 = 类名，值 = 类本身
        """延迟导入：库没装时不让整个模块 import 失败。"""
        try:
            from llama_index.core import (                    # 新 API 入口（llama-index >= 0.10）
                VectorStoreIndex, SimpleDirectoryReader, StorageContext,
                load_index_from_storage, Settings,
            )
            from llama_index.core.vector_stores import (
                MetadataFilters, MetadataFilter, FilterOperator,
            )
            return {                                          # 打包成字典：后续方法通过 self._li["类名"] 取用
                "VectorStoreIndex": VectorStoreIndex, "SimpleDirectoryReader": SimpleDirectoryReader,
                "StorageContext": StorageContext, "load_index_from_storage": load_index_from_storage,
                "Settings": Settings, "MetadataFilters": MetadataFilters,
                "MetadataFilter": MetadataFilter, "FilterOperator": FilterOperator,
            }
        except ImportError as e:                              # 新 API 路径不可用
            raise RAGException(                               # 转为 RAGException：工厂统一捕获后降级
                f"llama-index 未安装或版本过旧（需 llama_index.core）: {e}。"
                f"安装：pip install -r requirements-extras.txt"
            )

    # ------------------------------------------------------------------
    # 【索引加载】有落盘就加载，没有就建空索引
    #   空索引必须先建出来：否则第一次 retrieve 会因为索引不存在而抛错，
    #   而"知识库为空"是正常的初始状态，不该是一个异常。
    # ------------------------------------------------------------------
    # 【什么】索引就绪保障
    # 【作用】保证 self._index 不是 None：有旧索引加载旧索引，没有就建空索引
    # 【为什么】llama-index 的 retrieve 在空索引上返回空列表是正常行为；索引未创建就抛异常会让"刚初始化、还没导文档"的场景报错，
    #   这不是异常——用户只是还没导知识库而已
    def _ensure_index(self):
        if self._index is not None:                         # 已加载过
            return self._index                              # 直接复用句柄
        li = self._li                                       # 本地别名：减少字典查找
        index_json = os.path.join(self.persist_dir, "docstore.json")  # 标志性落盘文件：存在说明之前建过索引
        if os.path.exists(index_json):                      # 之前落盘过
            storage = li["StorageContext"].from_defaults(persist_dir=self.persist_dir)  # 从目录加载 storage
            self._index = li["load_index_from_storage"](storage)  # 从 storage 还原索引
        else:                                               # 首次使用或目录被清空
            self._index = li["VectorStoreIndex"].from_documents([])  # 用空文档创建空索引
            os.makedirs(self.persist_dir, exist_ok=True)    # 建落盘目录：为后续 persist 做准备
        return self._index                                  # 返回已就绪的索引句柄

    # 【什么】权限过滤条件翻译
    # 【作用】把 parse_expr 的中立结构转成 llama-index 的 MetadataFilters 对象
    # 【为什么】自研链路的 expr 是给 Milvus 用的；适配层必须翻译成 llama-index 的格式，
    #   否则权限过滤在 llama-index 后端失效 = 越权
    def _filters(self, expr: Optional[str]):
        """把权限 expr 翻译成 llama-index 的 MetadataFilters。"""
        parsed = parse_expr(expr)                           # 先走中立解析
        if not parsed:                                      # 没有过滤条件
            return None                                     # 返回 None = 不设过滤（与自研链路 expr=None 行为一致）
        li = self._li                                       # 本地别名
        op_map = {"in": li["FilterOperator"].IN, "==": li["FilterOperator"].EQ,
                  "!=": li["FilterOperator"].NE}           # 中立操作符 → llama-index 枚举的映射
        return li["MetadataFilters"](filters=[              # 组装过滤条件列表
            li["MetadataFilter"](key=p["key"], value=p["value"], operator=op_map[p["op"]])  # 每个条件转一个 MetadataFilter
            for p in parsed                                 # 逐条中立条件翻译
        ])

    # 【kb_code 为什么收下但不用】llama-index 后端是独立落盘的单一索引，没有分库概念；
    #   加这个参数只为与自研 RAGChain.ingest_document 签名一致，调用方才不用区分后端。
    # 【什么】单文档入库
    # 【作用】把文件写入 llama-index 索引并落盘
    # 【为什么】SimpleDirectoryReader 本身做内部切片；kb_code 参数仅为签名对齐——
    #   因为 llama-index 后端是单一索引，无法像 Milvus 那样按 kb_code 分物理集合
    def ingest_document(self, file_path: str, kb_code: str = "public") -> int:
        """文档入库：返回切片数（与自研 RAGChain.ingest_document 同签名）。"""
        docs = self._li["SimpleDirectoryReader"](input_files=[file_path]).load_data()  # llama-index 自带加载器：支持格式比自研 loader 多，但也受限于它自己的解析
        if not docs:                                        # 读不到内容
            _logger.warning(f"llama-index 文档无内容: {file_path}")  # 警告日志：通常意味着文件空或格式不被 SimpleDirectoryReader 识别
            return 0                                        # 返回 0 表示未入库
        index = self._ensure_index()                        # 确保索引已就绪
        for d in docs:                                      # 逐文档插入索引
            index.insert(d)                                 # 插入单篇文档
        index.storage_context.persist(persist_dir=self.persist_dir)  # 落盘：把索引持久化到磁盘，下次启动能恢复
        _logger.info(f"llama-index 入库完成: {file_path} | 文档={len(docs)}")  # 入库结果日志
        return len(docs)                                    # 返回入库文档数（自研链路返回的是分片数，这里语义略有差异但签名一致）

    # 【什么】目录批量入库
    # 【作用】把目录下所有文件读进索引并落盘
    # 【为什么】签名与自研 RAGChain.ingest_directory 对齐，上层调用方无感知
    def ingest_directory(self, dir_path: str) -> int:
        """批量入库（与自研 RAGChain.ingest_directory 同签名）。"""
        docs = self._li["SimpleDirectoryReader"](input_dir=dir_path).load_data()  # 递归加载目录下所有支持的文件
        index = self._ensure_index()                        # 确保索引已就绪
        for d in docs:                                      # 逐文档插入
            index.insert(d)                                 # 插入
        index.storage_context.persist(persist_dir=self.persist_dir)  # 落盘
        _logger.info(f"llama-index 批量入库完成: 共 {len(docs)} 文档")  # 汇总日志
        return len(docs)                                    # 返回总文档数

    # 【什么】检索
    # 【作用】从 llama-index 索引召回相关文档，字段名与自研链路保持一致
    # 【为什么】返回字段必须对齐（content / score / file_name / source / kb_code / updated_at），
    #   否则上层 build_context 会 KeyError —— 换后端最常踩的坑就是字段名/结构不一致
    def retrieve(self, query: str, top_k: Optional[int] = None, expr: Optional[str] = None) -> list:
        """
        检索：返回字段与自研链路保持一致（content/score/file_name/source），
        否则上层 build_context 会 KeyError —— 换后端最常踩的就是这个坑。
        """
        index = self._ensure_index()                        # 确保索引就绪
        retriever = index.as_retriever(
            similarity_top_k=top_k or self._top_k, filters=self._filters(expr)  # 取索引的检索器：传入 top_k 和权限过滤条件
        )
        nodes = retriever.retrieve(query)                   # 执行检索，返回 NodeWithScore 列表
        hits = []                                           # 收集整理后的结果
        for n in nodes:                                     # 逐条节点处理
            meta = getattr(n.node, "metadata", {}) or {}    # 取节点元数据：部分版本可能返回 None，用 or {} 兜底
            hits.append({                                    # 构造与自研链路字段对齐的字典
                "content": n.node.text,                     # 正文
                "score": float(n.score or 0.0),             # 相似度分数：llama-index 给分机制与 Milvus 不同，这里统一转成 float
                "file_name": meta.get("file_name", meta.get("file_path", "未知")),  # 文件名：取 file_name 优先，没有则取 file_path，再没有标"未知"
                "source": meta.get("file_path", ""),        # 来源路径
                "kb_code": meta.get("kb_code", ""),        # 知识库编码：llama-index 默认没有分库，除非写入时显式写进 metadata
                "updated_at": meta.get("updated_at", ""),  # 更新时间
            })
        return hits[:self._rerank_top_k] if self._rerank_top_k else hits  # 截断到 rerank_top_k：适配层没有接入 BGE 精排，直接用 similarity_top_k 截断

    # 【什么】检索 + 上下文拼接
    # 【作用】把 retrieve 的结果拼成"【知识1】(相关度:x 来源:y)\n正文"的文本块
    # 【为什么】输出格式必须与自研 RAGChain.build_context 一致：上层 prompt 模板只认这种格式，改了就拼不出上下文
    def build_context(self, query: str, top_k: Optional[int] = None, expr: Optional[str] = None) -> str:
        """拼接上下文（与自研 RAGChain.build_context 同格式，保证上层无感）。"""
        docs = self.retrieve(query, top_k=top_k, expr=expr)  # 先走检索
        if not docs:                                        # 空结果
            return ""                                       # 返回空串：上层据此判断"无可用知识"
        return "\n\n".join(                                  # 知识块之间空行分隔
            f"【知识{i}】(相关度:{d['score']:.2f} 来源:{d['file_name']})\n{d['content']}"  # 拼成带编号/分数/来源的知识块
            for i, d in enumerate(docs, 1)                 # 从 1 开始编号
        )

    # 【什么】知识库统计
    # 【作用】返回与自研链路同字段的统计信息
    # 【为什么】kb_codes 参数仅为接口对齐：llama-index 后端是单一索引，无法按库码分计，所以忽略该参数
    def get_knowledge_stats(self, kb_codes: Optional[list] = None) -> dict:
        """知识库统计（字段对齐自研 RAGChain.get_knowledge_stats）。

        kb_codes 仅为对齐接口签名：llama-index 后端是独立落盘的单一索引，
        没有分库概念，这里忽略该参数。
        """
        return {
            "backend": "llamaindex",                        # 后端标识：管理员一眼知道当前用的什么引擎
            "persist_dir": self.persist_dir,                # 落盘目录：运维查数据量/备份时直接看这里
            "index_ready": self._index is not None,         # 索引是否已加载：探活接口可用它判断就绪状态
            "total_chunks": self._chunk_count(),            # 总分片数（= 节点数）：自研链路返回 total_chunks，这里对齐字段名
            "by_kb": {},                                    # 分库明细：无法提供（单一索引），返回空字典表示该字段无意义
        }

    # 【什么】节点计数
    # 【作用】读取索引里实际存储的文档节点数
    # 【为什么】索引未就绪时不抛异常：统计接口只查个数，不该因为"还没建索引"导致整个管理页面报错
    def _chunk_count(self) -> int:
        """索引里的分片总数；索引未就绪时返回 0（不抛异常，统计接口不该拖垮页面）。"""
        try:
            return len(self._index.docstore.docs) if self._index is not None else 0  # 索引就绪时数节点数，未就绪返回 0
        except Exception:                                   # docstore 结构可能因版本差异不存在 docs 属性
            return 0                                        # 任何异常都返回 0：统计接口兜底


# 【什么】RAG 后端工厂
# 【作用】按配置/偏好返回自研 RAGChain 或适配层 LlamaIndexRAG 的实例
# 【为什么】上层始终拿到"有 retrieve/build_context 方法的对象"，不关心内部实现；
#   显式配置后端而不是默认切：因为 llama-index 的索引是独立落盘，切过去等于知识库空了，属于数据迁移级操作
def build_rag(prefer: str = ""):
    """
    RAG 后端工厂：按配置返回实现，上层拿到的都是"有 retrieve/build_context 的对象"。

    【为什么不默认切到 llama-index】
        它的索引是独立落盘的（persist_dir），切过去 = 现有 Milvus 知识库看不到，
        表现得像"知识库空了"。所以默认 native，只有显式配置才切 —— 后端切换
        属于数据迁移级别的操作，不能靠一个默认值悄悄发生。
    """
    backend = (prefer or get_settings().rag_backend or "native").lower()  # 后端名称来源：传入偏好 > 全局配置 > 默认 native
    if backend == "llamaindex":                             # 用户显式指定了 llama-index
        if not is_available():                              # 库不可用
            _logger.warning("RAG_BACKEND=llamaindex 但库不可用，降级为自研链路")  # 警告日志：提醒运维去装依赖
            backend = "native"                              # 降级到自研链路
        else:                                               # 库可用
            return LlamaIndexRAG()                          # 返回适配层实例
    from modules.rag.rag_chain import RAGChain              # 延迟导入：避免构造时的循环依赖
    return RAGChain()                                       # 默认返回自研链路实例
