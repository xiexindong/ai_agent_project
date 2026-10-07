"""
Milvus 向量存储模块
====================
静态知识库向量库：文档向量化、入库、相似度召回。
覆盖边界1（RAG无结果降级）、边界5（知识库过期管理）。

【两种部署形态：单集合 / 多集合】
    单集合（MILVUS_MULTI_COLLECTION=false，默认）：
        所有知识库共用一个集合（MILVUS_COLLECTION），库之间靠 kb_code 标量字段区分。
        内存开销小、跨库检索一次 ANN 搞定，行为与改造前完全一致。

    多集合（MILVUS_MULTI_COLLECTION=true）：
        每个知识库（kb_code）一个物理集合，集合名 = MILVUS_COLLECTION_PREFIX + kb_code，
        例如 kb_code="finance" → 集合 "kb_finance"。
        优点：库之间物理隔离，可整库 drop / 单独备份迁移 / 单独建索引。
        代价：每个集合都要 load() 常驻内存才能检索，库越多内存占用越大；
              跨库检索要对每个集合各搜一次再合并（本模块已封装，上层无感）。

    【为什么用开关而不是直接切】
        切换形态属于数据迁移级别的操作：打开开关后，旧集合里的数据不会自动搬过去，
        表现得像"知识库空了"。所以用配置显式开启，默认保持原行为。
"""
import re
from typing import List, Optional
from datetime import datetime
from modules.common.logger import get_logger
from modules.common.exceptions import RAGException, handle_exception
from modules.memory.embedding_service import get_embedding_service
from modules.rag.text_splitter import TextChunk
from config import get_settings

_logger = get_logger("milvus")


class MilvusVectorStore:
    """Milvus 向量数据库管理器（支持单集合与多集合两种形态）。"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        s = get_settings()

        self._default_collection_name = s.milvus_collection

        # 兼容旧代码/外部直接读 _collection_name 的写法
        self._collection_name = s.milvus_collection

        self._multi = bool(s.milvus_multi_collection)
        self._prefix = s.milvus_collection_prefix
        self._default_kb = s.milvus_default_kb

        self._dim = s.embedding_dimension

        self._top_k = s.rag_top_k

        self._threshold = s.similarity_threshold

        # 集合名 → Collection 对象（多集合模式下懒加载，用到哪库才 load 哪库，
        # 避免启动时把所有知识库一次性灌进内存）
        self._collections: dict = {}

        self._connect()

    def _connect(self):
        """连接 Milvus 并确保默认集合存在。"""
        try:
            from pymilvus import (
                connections, utility, Collection,
                CollectionSchema, FieldSchema, DataType,
            )
            s = get_settings()

            connections.connect(
                alias="default",
                host=s.milvus_host,
                port=str(s.milvus_port),
            )

            self._connections = connections
            self._utility = utility
            self._Collection = Collection
            self._DataType = DataType
            self._FieldSchema = FieldSchema
            self._CollectionSchema = CollectionSchema
            # 默认集合始终就绪，保证单集合模式与旧行为一致
            self._collection = self._ensure_collection(self._default_collection_name)
            _logger.info(
                f"Milvus连接成功: {s.milvus_host}:{s.milvus_port} | "
                f"模式={'多集合' if self._multi else '单集合'}"
            )
        except Exception as e:
            _logger.error(f"Milvus连接失败: {e}")
            raise RAGException("Milvus连接失败", str(e))

    # ------------------------------------------------------------------
    # 集合路由：kb_code ↔ 集合名
    # ------------------------------------------------------------------
    def collection_name_for(self, kb_code: Optional[str]) -> str:
        """
        知识库编码 → Milvus 集合名。

        单集合模式一律返回默认集合（kb_code 只作为标量字段写入）；
        多集合模式返回 前缀 + kb_code，如 "finance" → "kb_finance"。
        """
        if not self._multi:
            return self._default_collection_name
        return f"{self._prefix}{kb_code or self._default_kb}"

    def kb_of_collection(self, collection_name: str) -> str:
        """集合名 → 知识库编码（collection_name_for 的逆运算）。"""
        if not self._multi:
            return self._default_kb
        return collection_name[len(self._prefix):] if collection_name.startswith(self._prefix) else collection_name

    def _ensure_collection(self, name: str):
        """确保指定集合存在（不存在则建 + 建索引），加载并返回。"""
        # ------------------------------------------------------------------
        # 【建表：Milvus 的"集合"≈ 关系库的"表"】
        #   一张表由若干字段组成：除了 embedding 向量列，还要带上 chunk_id/content/
        #   kb_code 等标量列——检索先按向量找到"最相近的行"，再把这些列读回来
        #   （正文内容、来源、属于哪个知识库）。
        #
        # 【为什么要这个？】集合不存在直接写数据会报错；所以先 has_collection 查，
        #   不存在才建。max_length 按该列最长内容预留：content 放整片给 65535，
        #   kb_code 只是短编码给 64。
        #
        # 【对比例子】文档一个分片入库后的一行：
        #   chunk_id="chk_42" | doc_id="doc_7" | content="报销需要提交发票..."
        #   kb_code="finance" | dept_code="D_FIN" | embedding=[0.01,-0.3,...共1024维]
        # ------------------------------------------------------------------
        if not self._utility.has_collection(name):
            fields = [
                # is_primary=True：分片唯一主键，Milvus 靠它去重/定位单行
                self._FieldSchema(name="chunk_id", dtype=self._DataType.VARCHAR, max_length=128, is_primary=True),
                self._FieldSchema(name="doc_id", dtype=self._DataType.VARCHAR, max_length=128),     # 所属文档 id，同文档多分片共用
                self._FieldSchema(name="content", dtype=self._DataType.VARCHAR, max_length=65535),  # 分片正文，给最大长度
                self._FieldSchema(name="source", dtype=self._DataType.VARCHAR, max_length=512),     # 来源路径/链接
                self._FieldSchema(name="file_name", dtype=self._DataType.VARCHAR, max_length=256),  # 上传文件名
                self._FieldSchema(name="updated_at", dtype=self._DataType.VARCHAR, max_length=32),  # 更新时间（存字符串）
                self._FieldSchema(name="kb_code", dtype=self._DataType.VARCHAR, max_length=64),     # 知识库编码：权限过滤靠它
                self._FieldSchema(name="dept_code", dtype=self._DataType.VARCHAR, max_length=64),   # 部门编码：部门级隔离用
                # 唯一的向量列：dim 必须与 embedding 模型输出维度一致（本项目 1024）
                self._FieldSchema(name="embedding", dtype=self._DataType.FLOAT_VECTOR, dim=self._dim),
            ]
            
            schema = self._CollectionSchema(fields, description="企业知识库RAG向量索引")  # 字段清单打包成表结构
            collection = self._Collection(name, schema=schema)    # 此时才真正在 Milvus 创建空集合
            # ------------------------------------------------------------------
            # 【建索引：给 embedding 列装"向量查找目录"】
            #   不建索引只能全表逐行算距离（十万行就很慢）；索引把向量预先分桶，
            #   查询只扫最可能命中的桶，用一点精度换大幅提速。
            #     index_type=IVF_FLAT：向量聚成 nlist 个桶，查时只扫最近的几个桶
            #     metric_type=COSINE ：余弦距离，和我们 embedding 的口径一致
            #     nlist=128          ：桶的数量，数据越多可调越大
            # ------------------------------------------------------------------
            collection.create_index(
                field_name="embedding",
                index_params={
                    "index_type": "IVF_FLAT",
                    "metric_type": "COSINE",
                    "params": {"nlist": 128},
                },
            )
            _logger.info(f"Milvus集合创建: {name}")
        collection = self._Collection(name)                        # 集合此刻必然存在（刚建或原本有），取句柄
        self._collections[name] = collection                       # 先进缓存
        collection.load()                                          # load 进内存才能被检索；不 load 查询会报错
        _logger.info(
            f"Milvus集合就绪: {name} | 知识库隔离字段={self._has_field(name, 'kb_code')}"  # 顺带报告老集合有没有权限字段
        )
        return collection

    def _get_collection(self, name: str):
        """取集合（带缓存，首次访问才创建并 load）。"""
        collection = self._collections.get(name)
        if collection is None:
            collection = self._ensure_collection(name)
        return collection

    def _has_field(self, name: str, field: str) -> bool:
        """判断集合有没有某个标量字段（老集合可能没有 kb_code，需要兼容）。"""
        try:
            return any(f.name == field for f in self._get_collection(name).schema.fields)
        except Exception:
            return False

    def _target_collections(self, kb_codes: Optional[list]) -> list:
        """把 kb_codes 转成要访问的集合名列表；为空时退回默认集合。"""
        if not kb_codes:
            return [self.collection_name_for(self._default_kb)] if self._multi else [self._default_collection_name]
        return [self.collection_name_for(c) for c in kb_codes]

    def list_knowledge_bases(self) -> list:
        """列出已存在的知识库编码。

        多集合模式：扫描 Milvus 里所有带前缀的集合名反推（如 kb_finance → finance）。
        单集合模式：只有一个逻辑库，返回默认库编码。
        """
        if not self._multi:
            return [self._default_kb]
        try:
            names = self._utility.list_collections() or []
        except Exception as e:
            _logger.warning(f"列举Milvus集合失败: {e}")
            return [self._default_kb]
        return [n[len(self._prefix):] for n in names if n.startswith(self._prefix)]

    @handle_exception(default=False, log_msg="Milvus知识库下线异常")
    def drop_kb(self, kb_code: str) -> bool:
        """整库下线：直接 drop 掉整个物理集合（多集合模式独有的能力）。

        单集合模式下没法这么做（会把别人的库一起删了），只能按 doc_id 下线文档。
        """
        if not self._multi:
            _logger.warning("单集合模式不支持整库 drop，请改用 remove_by_doc_id 按文档下线")
            return False
        name = self.collection_name_for(kb_code)
        if self._utility.has_collection(name):
            self._utility.drop_collection(name)
            self._collections.pop(name, None)
            _logger.info(f"知识库已整库删除: kb={kb_code} | 集合={name}")
        return True

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def insert_chunks(self, chunks: List[TextChunk], kb_code: Optional[str] = None) -> int:
        """
        将文本分片向量化后入库。

        Args:
            chunks: 文本分片
            kb_code: 目标知识库。为空时取 chunk.metadata["kb_code"]，再没有就用默认库。
                     一批 chunks 里 kb_code 不同也没关系，会按目标集合自动分组写入。
        """
        if not chunks:
            return 0
        emb_service = get_embedding_service()
        vectors = emb_service.embed_batch([c.content for c in chunks])
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 按目标集合分组：多集合模式下不同 kb_code 落进不同物理集合
        groups: dict = {}
        for i, c in enumerate(chunks):
            code = kb_code or c.metadata.get("kb_code") or self._default_kb
            groups.setdefault(self.collection_name_for(code), []).append((i, code, c))

        total = 0
        for name, items in groups.items():
            collection = self._get_collection(name)
            data = [
                [c.metadata.get("chunk_id", f"chunk_{i}") for i, _, c in items],
                [c.metadata.get("doc_id", "") for _, _, c in items],
                [c.content for _, _, c in items],
                [c.metadata.get("source", "") for _, _, c in items],
                [c.metadata.get("file_name", "") for _, _, c in items],
                [now] * len(items),
            ]
            if self._has_field(name, "kb_code"):
                data.append([code for _, code, _ in items])
                data.append([c.metadata.get("dept_code", "") for _, _, c in items])
            data.append([vectors[i] for i, _, _ in items])
            collection.insert(data)
            collection.flush()
            total += len(items)
            _logger.info(f"向量入库完成: 集合={name} | {len(items)} 条分片")
        return total

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    @handle_exception(default=None, log_msg="Milvus向量召回异常", reraise=False)
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        expr: Optional[str] = None,
        kb_codes: Optional[list] = None,
    ) -> list[dict]:
        """
        向量相似度召回，返回 [{content, source, file_name, score, kb_code, collection, ...}]。
        相似度低于 threshold 的结果会被过滤。

        Args:
            expr: Milvus 标量过滤表达式，用于知识库权限隔离。
                  例：'kb_code in ["public","finance"]' → 只在公开+财务库里搜。
                  None = 不限制。老集合无 kb_code 字段时会自动回退。
            kb_codes: 显式指定检索哪些知识库，优先级高于 expr 里的 kb_code 条件。
                      RBAC 场景下可直接传 principal.kb_codes，免去拼表达式。

        【多集合模式下的跨库检索】
            kb_code 已经体现在集合名上，所以逐库各搜一次、合并后按 score 重排取 top_k
            ——单个集合的 ANN 只在自己库内生效，不合并就会漏掉其他库的更优结果。
        """
        emb_service = get_embedding_service()
        query_vec = emb_service.embed(text=query)
        k = top_k or self._top_k

        # ------------------------------------------------------------------
        # 【拆分 expr】一条权限表达式在这里被劈成两半，分别驱动"搜哪个集合"和"集合内怎么过滤"
        #
        #   【处理前】expr = 'kb_code in ["public","finance"] and dept_code == "D_FIN"'
        #            （由 RBACService.kb_expr(principal) 生成，见 rbac_service.py:326）
        #            另可传 kb_codes=['public','finance']，优先级高于 expr 里的 kb_code 条件
        #
        #   【处理后】
        #     codes      = ['public', 'finance']        ← 库名列表 → 下一行决定搜哪几个集合
        #     inner_expr = 'dept_code == "D_FIN"'       ← 去掉 kb_code 后的残余条件
        #                                                 （剥空后残留的 and/or 也被清掉，
        #                                                  全剥完则返回 None = 集合内不加过滤）
        #
        #   【为什么要劈开】多集合模式下 kb_code 已体现在集合名上（kb_finance），
        #     集合里再过滤 kb_code 既多余、在老集合上还会因缺字段报错；
        #     所以 kb_code 条件翻译成"路由到哪几个集合"，其余条件才留给集合内过滤。
        #
        #   【实测对照表】_parse_kb_codes(expr) / _strip_kb_expr(expr)
        #     'kb_code in ["public","finance"]'                        → ['public','finance'] | None
        #     'kb_code in ["public","finance"] and dept_code == "D_FIN"' → ['public','finance'] | 'dept_code == "D_FIN"'
        #     'dept_code == "D_FIN" and kb_code in ["hr"]'             → ['hr'] | 'dept_code == "D_FIN"'
        #     'updated_at > "2026-01-01"'（无 kb_code）                 → [] | 原样返回
        #     None / ''                                                → [] | None（不限制）
        #
        #   【两种模式的取值差异】
        #     单集合（_multi=False）：inner_expr = expr 原样 —— kb_code 是标量字段，必须过滤
        #     多集合（_multi=True） ：inner_expr = 剥离后的残余条件 —— kb_code 由集合名承担
        #
        #   【易踩边界】
        #     ① 只认 `kb_code in [...]` 写法；写成 kb_code == "finance" 解析不出 codes，
        #        且 strip 不会剥掉它 → 多集合下拿不存在的字段过滤 → 触发 _search_in 的
        #        无过滤回退（308-312 行）
        #     ② 'kb_code in []' → codes=[] → _target_collections([]) 退回默认知识库，
        #        不等于"什么都搜不到"；想表达无可见库应让 kb_expr 返回 None 并由上层拒绝
        # ------------------------------------------------------------------
        # 【什么】把 expr 拆成「搜哪几个库」+「库内过滤条件」两个产物
        # 【作用】让同一条 RBAC 表达式在单集合 / 多集合两种模式下都不用改就能生效
        codes = list(kb_codes) if kb_codes else self._parse_kb_codes(expr)   # 库名：优先用显式 kb_codes，否则从 expr 反解析
        inner_expr = self._strip_kb_expr(expr) if self._multi else expr      # 库内条件：多集合剥掉 kb_code，单集合原样保留
        targets = self._target_collections(codes)

        hits = []
        for name in targets:
            try:
                hits.extend(self._search_in(name, query_vec, k, inner_expr))
            except Exception as e:
                # 某一个库挂了不能让整个检索失败（比如该集合还没建索引）
                _logger.warning(f"集合 {name} 检索失败，已跳过: {e}")

        hits.sort(key=lambda h: h["score"], reverse=True)
        hits = hits[:k]
        _logger.info(
            f"向量召回: query='{query[:30]}...' | 命中={len(hits)}/{k} | "
            f"集合={[t for t in targets]} | 知识库过滤={'有' if (expr or kb_codes) else '无'}"
        )
        return hits

    def _search_in(self, name: str, query_vec: list, k: int, expr: Optional[str]) -> list:
        """在单个集合里做一次 ANN 检索并整理结果。"""
        collection = self._get_collection(name)
        base_fields = ["chunk_id", "doc_id", "content", "source", "file_name", "updated_at"]
        extra_fields = ["kb_code", "dept_code"] if self._has_field(name, "kb_code") else []
        try:
            results = self._search_once(collection, query_vec, k, expr, base_fields + extra_fields)
        except Exception as e:
            if not expr:
                raise
            _logger.warning(f"集合 {name} 过滤检索失败，回退无过滤（疑似集合缺 kb_code 字段）: {e}")
            results = self._search_once(collection, query_vec, k, None, base_fields)

        kb = self.kb_of_collection(name)
        hits = []
        # pymilvus 的 search 返回类型被标注成 Future/Coroutine 联合，实际是同步结果列表
        for hit in results[0]:   # type: ignore
            score = hit.score
            if score < self._threshold:
                _logger.debug(f"低相关结果过滤: score={score:.4f} < {self._threshold}")
                continue
            hits.append({
                "content": hit.entity.get("content", ""),
                "source": hit.entity.get("source", ""),
                "file_name": hit.entity.get("file_name", ""),
                "doc_id": hit.entity.get("doc_id", ""),
                "updated_at": hit.entity.get("updated_at", ""),
                "score": round(score, 4),
                # 单集合模式下 kb_code 来自字段值；多集合模式下来自集合名
                "kb_code": hit.entity.get("kb_code", kb) if "kb_code" in extra_fields else kb,
                "collection": name,
            })
        return hits

    @handle_exception(default=[], log_msg="Milvus关键词降级检索异常")
    def keyword_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        expr: Optional[str] = None,
        kb_codes: Optional[list] = None,
    ) -> list[dict]:
        """边界1降级通道：向量无结果时，用 Milvus 的 like 表达式做关键词粗召回。

        命中结果的 score 统一置为 threshold，交给下游 reranker 用关键词分重排，
        避免"降级结果分数为 0 又被阈值过滤掉"的二次丢失。
        """
        keywords = [
            w for w in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]{2,}", query)
        ]
        if not keywords:
            return []
        keywords.sort(key=len, reverse=True)
        keywords = keywords[:3]

        conds = " or ".join(f'content like "%{kw}%"' for kw in keywords)

        codes = list(kb_codes) if kb_codes else self._parse_kb_codes(expr)
        inner_expr = self._strip_kb_expr(expr) if self._multi else expr
        targets = self._target_collections(codes)

        limit = top_k or self._top_k
        hits = []
        for name in targets:
            final_expr = f"({conds})" + (f" and {inner_expr}" if inner_expr else "")
            try:
                rows = self._get_collection(name).query(
                    expr=final_expr,
                    output_fields=["chunk_id", "doc_id", "content", "source", "file_name", "updated_at"],
                    limit=limit,
                )
            except Exception as e:
                _logger.warning(f"集合 {name} 关键词检索失败，已跳过: {e}")
                continue
            kb = self.kb_of_collection(name)
            hits.extend({
                "content": r.get("content", ""),
                "source": r.get("source", ""),
                "file_name": r.get("file_name", ""),
                "doc_id": r.get("doc_id", ""),
                "updated_at": r.get("updated_at", ""),
                "score": self._threshold,
                "kb_code": kb,
                "collection": name,
                "recall_type": "keyword",
            } for r in rows)   # type: ignore

        hits.sort(key=lambda h: h["score"], reverse=True)
        hits = hits[:limit]
        _logger.info(f"关键词降级召回: 关键词={keywords} | 命中={len(hits)}")
        return hits

    @staticmethod
    def _parse_kb_codes(expr: Optional[str]) -> list:
        """从 'kb_code in ["public","finance"]' 里提取出 ['public','finance']。

        这样 RBAC 生成的表达式不用改就能驱动多集合路由：
        表达式里的 kb_code 条件在多集合模式下被翻译成"搜哪几个集合"。
        """
        if not expr:
            return []
        m = re.search(r'kb_code\s+in\s*\[(.*?)\]', expr)
        if not m:
            return []
        return [x.strip().strip('"\'') for x in m.group(1).split(",") if x.strip()]

    @staticmethod
    def _strip_kb_expr(expr: Optional[str]) -> Optional[str]:
        """去掉表达式里的 kb_code in [...] 片段，留下集合内的其他过滤条件。

        'kb_code in ["a","b"] and dept_code == "D_FIN"' → 'dept_code == "D_FIN"'
        """
        if not expr:
            return None
        m = re.search(r'kb_code\s+in\s*\[.*?\]', expr)
        if not m:
            return expr
        rest = expr.replace(m.group(0), "")
        rest = re.sub(r'^\s*(and|or)\s*', '', rest, flags=re.I).strip()
        rest = re.sub(r'\s*(and|or)\s*$', '', rest, flags=re.I).strip()
        return rest or None

    def _search_once(self, collection, query_vec: list, k: int, expr: Optional[str], output_fields: list):
        """执行一次 ANN 检索；expr 为空时不下过滤条件（兼容老集合）。"""
        kwargs = dict(
            # 查询向量：包一层 list 变成"批次大小=1"——search 接口按批收向量，
            #   返回结果也按批给，取结果时固定用 hits[0] 取第一组
            data=[query_vec],
            # 在哪个向量列上搜：必须与建集合/插入时的字段名一致（见 ensure_collection 的 embedding 列）
            anns_field="embedding",
            # 检索参数：
            #   metric_type=COSINE 余弦相似度——必须与建索引时一致，换度量等于换尺子，
            #     查出来的分数全部失真；
            #   nprobe=16 搜 128 个 IVF 聚类里的前 16 个桶（nlist=128）——
            #     调大→召回更高但更慢，调小→更快但容易漏。16 是"准确优先"的折中
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            # 只要 top-k 条（k 由上层传入：召回层与重排层各用不同 k）
            limit=k,
            # 附带返回这些标量列（doc_id/chunk_text/kb_code 等），
            #   免得拿 id 再回查一次库
            output_fields=output_fields,
        )
        # 【为什么 expr 不放进上面的 dict】Milvus 的 expr=None 与"不传"不等价：
        #   显式传 None 在部分版本会触发校验报错。只在真的有过滤条件时才加 key，
        #   老集合（没有 kb_code 列的）不带过滤也能正常搜——这就是 docstring 说的"兼容老集合"
        if expr:
            kwargs["expr"] = expr
        return collection.search(**kwargs)

    # ------------------------------------------------------------------
    # 删除与统计
    # ------------------------------------------------------------------
    @handle_exception(default=False, log_msg="Milvus文档下线异常")
    def remove_by_doc_id(self, doc_id: str, kb_code: Optional[str] = None) -> bool:
        """按文档ID下线过期文档（边界5：知识库过期管理）。

        kb_code 为空且处于多集合模式时，会在所有知识库集合里逐个删。
        """
        expr = f'doc_id == "{doc_id}"'
        if not self._multi:
            names = [self._default_collection_name]
        elif kb_code:
            names = [self.collection_name_for(kb_code)]
        else:
            names = [self.collection_name_for(k) for k in self.list_knowledge_bases()]

        for name in names:
            collection = self._get_collection(name)
            collection.delete(expr)
            collection.flush()
        _logger.info(f"文档下线完成: doc_id={doc_id} | 涉及集合={names}")
        return True

    @handle_exception(default=0, log_msg="Milvus统计异常")
    def count(self, kb_codes: Optional[list] = None) -> int:
        """统计向量总数。kb_codes 为空 = 统计全部知识库（单集合模式即整个集合）。"""
        return sum(self.count_by_kb(kb_codes).values())

    @handle_exception(default={}, log_msg="Milvus分库统计异常")
    def count_by_kb(self, kb_codes: Optional[list] = None) -> dict:
        """按知识库分别统计：{"public": 120, "finance": 45}

        多集合模式下每个库是独立集合，天然能分开数；
        单集合模式下只有一个逻辑库，返回 {默认库: 总数}。
        """
        names = self._target_collections(kb_codes)
        out = {}
        for name in names:
            try:
                collection = self._get_collection(name)
                # 把内存缓冲区（growing segment）里刚写入的向量刷成已持久化段。
                # 不 flush 的话 num_entities 只数已落盘的数据，刚入库的文档会被漏掉 → 计数偏小。
                # 代价：flush 是同步阻塞调用，会等落盘完成，高频调用会拖慢写入吞吐。
                collection.flush()
                out[self.kb_of_collection(name)] = collection.num_entities
            except Exception as e:
                _logger.warning(f"集合 {name} 统计失败: {e}")
                out[self.kb_of_collection(name)] = 0
        return out
