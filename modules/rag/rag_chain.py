"""
RAG 链路模块
============
整合文档加载 → 切片 → 向量化 → 入库 → 召回 → 精排 → 上下文拼接的完整 RAG 闭环。
覆盖边界1（RAG无结果/低相关降级处理）。
"""
from typing import List, Optional                           # 类型标注：List 列表 / Optional 可选参数
from datetime import datetime                               # 时间处理：recency_boost 按文档更新时间加权用
from modules.common.logger import get_logger                # 统一日志工厂
from modules.common.exceptions import RAGException          # RAG 业务异常类型（本模块预留，子模块抛出的同类异常会向上传）
from modules.rag.document_loader import DocumentLoader, Document  # 文档加载器 + 文档数据结构
from modules.rag.text_splitter import TextChunk, TextSplitter     # 分片数据结构 + 切分器
from modules.rag.vector_store import MilvusVectorStore      # Milvus 向量库：向量化入库与召回
from modules.rag.reranker import BGEReranker                # BGE 精排器：召回结果的二次重排
from config import get_settings                             # 全局配置读取

_logger = get_logger("rag_chain")                           # 本模块日志器，命名 rag_chain，便于日志定位来源


# 【什么】RAG 全链路管理器
# 【作用】把"加载→切片→入库→检索→精排→拼上下文"串成一条闭环，对上层（ReActAgent / API）只暴露
#   ingest_document / retrieve / build_context 等少数方法
# 【为什么】编排逻辑收敛在这一层：各子模块（loader/splitter/store/reranker）互不依赖、可单测，
#   上层不用知道 RAG 内部有几步、每步谁来做
class RAGChain:
    """RAG 全链路管理器。"""

    # 【什么】构造器：组装链路各环节
    # 【作用】创建加载器、切分器、向量库、精排器四个部件，并快照检索相关配置
    # 【为什么】部件在构造时一次性装好：运行期行为可预期；配置快照避免运行中改配置导致行为漂移
    def __init__(   
        self,
        chunk_size: int = 500,                              # 切分粒度：每片最大 500 字符（语义完整与检索精准的折中）
        chunk_overlap: int = 80,                            # 相邻片重叠 80 字符（跨片句子不断裂）
    ):
        self.loader = DocumentLoader()                      # 第 1 环：文档加载器（TXT/MD/PDF → Document）
        self.splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)  # 第 2 环：切分器（Document → TextChunk 列表）
        self.vector_store = MilvusVectorStore()             # 第 3 环：向量库（分片向量化入库 + 相似度召回），单例
        self.reranker = BGEReranker()   
        self._rerank_top_k = get_settings().rerank_top_k    # 精排后保留的条数（最终进上下文的条数）
        self._top_k = get_settings().rag_top_k              # 向量召回条数（粗筛条数，通常大于精排条数）
        self._keyword_fallback = get_settings().rag_keyword_fallback  # 边界1开关：向量无结果时是否启用关键词降级检索
        self._recency_boost = get_settings().rag_recency_boost      # 边界5开关：是否按文档新旧给召回分加权

    # 【什么】单文档入库全流程
    # 【作用】加载 → 切片 → 向量化 → 写入 Milvus，返回入库分片数
    # 【为什么】四步串成一个方法是给上层一个"扔路径就完事"的一站式入口；
    #   空文档返回 0 而不是抛错：批量入库时空文件不该中断整批
    def ingest_document(self, file_path: str, kb_code: Optional[str] = None) -> int:
        """
        文档入库全流程：加载 → 切片 → 向量化 → 写入Milvus。
        返回入库分片数。

        Args:
            kb_code: 目标知识库编码。多集合模式下决定写进哪个物理集合；
                     为空时由 chunk 的 metadata 决定，再没有则落到默认库。
        """
        doc = self.loader.load(file_path)                   # 第 1 步：加载文件成 Document（格式自动识别）
        # 【切片】TextSplitter.split_document() → modules/rag/text_splitter.py:95
        #   按 chunk_size=500 / chunk_overlap=80 切分（29 行构造时传入），
        #   每个分片带上 chunk_id/chunk_index/total_chunks 元数据，供检索后溯源与拼接；
        #   返回 List[TextChunk]，空列表 = 文档无可切内容（见下面 52 行的判断）
        chunks: List[TextChunk] = self.splitter.split_document(doc)  # 第 2 步：把文档切成带元数据的分片
        if not chunks:                                      # 切不出任何分片（空文档 / 全空白）
            _logger.warning(f"文档无有效内容: {file_path}")  # 记警告：这通常意味着文件坏了或真是空的
            return 0                                        # 返回 0 表示什么都没入库，不抛错
        # 【向量化 + 写库】MilvusVectorStore.insert_chunks() → modules/rag/vector_store.py:207
        #   内部三步：embed_batch 批量向量化 → 按 kb_code 分组到不同物理集合 → insert + flush
        #   kb_code 为空时取 chunk.metadata["kb_code"]，再没有就落默认库（215 行注释）
        #   返回实际写入条数；本行没接返回值，下面日志用的是 len(chunks)（切出多少就报多少）
        self.vector_store.insert_chunks(chunks, kb_code=kb_code)  # 第 3 步：分片批量向量化并写入 Milvus
        _logger.info(f"文档入库完成: {file_path} | 分片={len(chunks)} | 知识库={kb_code or '默认'}")  # 入库结果日志
        return len(chunks)                                  # 返回入库分片数

    # 【什么】目录批量入库
    # 【作用】把目录下所有支持的文档逐个走"加载→切片→入库"流程
    # 【为什么】知识库初始化 / 批量导入场景入口；单文档空内容跳过（if chunks），不中断整批
    def ingest_directory(self, dir_path: str, kb_code: Optional[str] = None) -> int:
        """批量入库目录下所有文档。"""
        docs = self.loader.load_batch(dir_path)             # 递归扫描目录，批量加载所有文档
        total = 0                                           # 累计入库分片数
        for doc in docs:                                    # 逐文档处理
            chunks = self.splitter.split_document(doc)      # 切片
            if chunks:                                      # 空文档跳过
                self.vector_store.insert_chunks(chunks, kb_code=kb_code)  # 向量化入库
                total += len(chunks)                        # 累加分片数
        _logger.info(f"批量入库完成: 共 {total} 分片 | 知识库={kb_code or '默认'}")  # 批量入库汇总日志
        return total                                        # 返回总入库分片数

    # 【什么】知识库检索
    # 【作用】Milvus 向量召回 → （可选）关键词降级 → （可选）新旧加权 → BGE 精排，返回 top_k
    # 【为什么】权限过滤（expr / kb_codes）必须在召回层生效而不是召回后剔除——
    #   后者等于把越权文档送进了模型上下文，泄密不可撤回（见 docstring）
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        expr: Optional[str] = None,
        kb_codes: Optional[list] = None,
    ) -> List[dict]:
        """
        知识库检索：Milvus向量召回 → BGE精排过滤。
        返回精排后的 top_k 相关文档列表。

        Args:
            expr: 知识库权限过滤表达式（由 RBACService.kb_expr 生成）。
                  None = 不限制。权限隔离在这里生效：越权的文档根本不会被召回，
                  而不是召回后再剔除——后者等于把机密内容送进了模型上下文。
            kb_codes: 显式指定检索哪些知识库（优先级高于 expr 里的 kb_code 条件）。
                      直接传 principal.kb_codes 即可，不必自己拼表达式。
        """
        k = top_k or self._top_k                            # 召回条数：没传用配置默认值
        candidates = self.vector_store.search(query, top_k=k, expr=expr, kb_codes=kb_codes)  # 第 1 步：Milvus 向量召回（权限过滤在这一步生效）

        # 边界1 降级设计：向量召回空结果时，用关键词 like 粗召回兜底——
        #  embedding 对生僻词/缩写可能完全失配，关键词匹配是最后的保命通道
        if not candidates and self._keyword_fallback:       # 向量无结果且开了降级开关
            candidates = self.vector_store.keyword_search(query, top_k=k, expr=expr, kb_codes=kb_codes)  # 走关键词降级召回
            if candidates:                                  # 降级通道有命中
                _logger.info(f"边界1降级：向量无结果，关键词检索补充命中 {len(candidates)} 条")  # 记录降级事件（监控关注）

        if not candidates:                                  # 两条通道都没命中
            _logger.warning(f"RAG无结果，触发兜底（边界1）: query='{query[:50]}'")  # 记警告：query 截断 50 字防日志爆炸
            return []                                       # 返回空列表：上层拼出空上下文，禁止模型幻觉编造

        if self._recency_boost:                             # 开了新旧加权开关（边界5）
            self._apply_recency_boost(candidates)           # 按更新时间给分：新文档加分，旧文档不加分

        reranked = self.reranker.rerank(query, candidates, top_k=self._rerank_top_k)  # 第 2 步：BGE 精排（模型不可用时内部自动降级）
        return reranked if reranked else []                 # rerank 异常返回 None 时兜底成空列表，保证返回类型稳定

    # 【什么】召回结果按新旧加权
    # 【作用】30 天内 +0.05、90 天内 +0.02，直接原地改 candidates 里的 score
    # 【为什么】边界5：知识库会过期——同名文档新旧两版都在库里时，不给新文档加分，
    #   旧版可能以微弱分差压过新版被采用（比如旧报销标准）。小幅度加分只影响"差不多相关"的排序，不颠覆语义排序
    @staticmethod                                           # 静态方法：纯函数式处理，不依赖实例状态
    def _apply_recency_boost(candidates: list) -> None:     # 原地修改候选列表，无返回值
        """边界5：按文档更新时间加权，30天内 +0.05、90天内 +0.02，避免旧版文档压过新版。"""
        now = datetime.now()                                # 取当前时间作为新旧基准
        for c in candidates:                                # 逐条候选处理
            try:
                days = (now - datetime.strptime(c.get("updated_at", ""), "%Y-%m-%d %H:%M:%S")).days  # 算出文档距今天数
            except Exception:                               # 时间字段缺失/格式不对（比如关键词降级通道的老数据）
                continue                                    # 解析不了就不加分，跳过本条
            bonus = 0.05 if days <= 30 else (0.02 if days <= 90 else 0.0)  # 阶梯加分：30 天内 0.05，90 天内 0.02，更久不加
            if bonus:                                       # 有加分的才改
                c["score"] = round(c.get("score", 0.0) + bonus, 4)  # 原地加分并保留 4 位小数

    # 【什么】检索 + 上下文拼接
    # 【作用】把 retrieve 的结果拼成"【知识1】(相关度:x 来源:y)\n正文"的文本块，直接喂给 LLM
    # 【为什么】无结果返回空字符串是硬约束：上层 prompt 拿到空上下文应明确说"知识库没有相关资料"，
    #   而不是让模型自由发挥——这是防幻觉的最后一道闸
    def build_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        expr: Optional[str] = None,
        kb_codes: Optional[list] = None,
    ) -> str:
        """
        检索知识库并拼接为上下文文本，供LLM问答使用。
        无结果时返回空字符串，禁止模型幻觉编造。
        """
        docs = self.retrieve(query, top_k=top_k, expr=expr, kb_codes=kb_codes)  # 先走完整检索链路
        if not docs:                                        # 一条都没命中
            return ""                                       # 返回空串：上层据此走"无资料"话术，禁止编造
        blocks = []                                         # 累积每条知识的文本块
        for i, doc in enumerate(docs, 1):                   # 从 1 开始编号
            score = doc.get("rerank_score", doc.get("score", 0))  # 优先展示精排分，没有精排分用召回分
            source = doc.get("file_name", doc.get("source", "未知"))  # 优先展示文件名，没有则用来源路径，再没有标"未知"
            blocks.append(f"【知识{i}】(相关度:{score:.2f} 来源:{source})\n{doc['content']}")  # 拼成带编号/分数/来源的知识块，模型可引用来源
        return "\n\n".join(blocks)                          # 知识块之间空行分隔后返回完整上下文

    # 【什么】知识库统计信息
    # 【作用】返回总分片数与分库明细，供管理后台/探活接口展示
    # 【为什么】kb_codes 参数是权限视角：普通用户只传自己可见的库，避免把"全公司文档总量"泄露给单库权限用户
    def get_knowledge_stats(self, kb_codes: Optional[list] = None) -> dict:
        """
        获取知识库统计信息。

        Args:
            kb_codes: 只统计这些知识库；为空 = 统计全部（管理员视角）。
                      传入当前用户可见的 kb_codes，可以避免把"全公司文档总量"
                      暴露给只有单个库权限的普通用户。

        Returns:
            {
                "total_chunks": 总分片数,
                "collection":   默认集合名（兼容旧字段）,
                "collections":  本次统计涉及的集合名列表,
                "by_kb":        {"public": 120, "finance": 45} 分库明细
            }
        """
        by_kb = self.vector_store.count_by_kb(kb_codes)     # 分库统计：{库码: 分片数}
        return {
            "total_chunks": sum(by_kb.values()),            # 总分片数 = 各库之和
            "collection": self.vector_store._collection_name,  # 默认集合名（保留旧字段，兼容老调用方）
            "collections": self.vector_store._target_collections(kb_codes),  # 本次统计实际涉及的物理集合名列表
            "by_kb": by_kb,                                 # 分库明细原样透出
        }
