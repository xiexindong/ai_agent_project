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
import re                                                   # 正则表达式：解析权限表达式里的 kb_code in [...] 片段
from typing import List, Optional                            # 类型标注
from datetime import datetime                               # 生成入库时间戳
from modules.common.logger import get_logger                # 统一日志工厂
from modules.common.exceptions import RAGException, handle_exception  # RAG 业务异常 + 异常兜底装饰器
from modules.memory.embedding_service import get_embedding_service  # 向量化服务：把文本转成 embedding 向量
from modules.rag.text_splitter import TextChunk             # 分片数据结构：insert_chunks 的入参类型
from config import get_settings                             # 全局配置读取

_logger = get_logger("milvus")                              # 本模块日志器，命名 milvus，便于日志定位来源


# 【什么】Milvus 向量数据库管理器
# 【作用】封装连接管理、集合路由（单集合/多集合）、写入、检索、删除、统计
# 【为什么】用单例模式：Milvus 连接和集合 load() 耗资源，重复创建会导致连接泄漏和重复 load 内存暴涨；
#   路由逻辑（kb_code ↔ 集合名）集中在这层，上层只管传 kb_code，不必理解单集合与多集合差异
class MilvusVectorStore:
    """Milvus 向量数据库管理器（支持单集合与多集合两种形态）。"""

    _instance = None                                        # 类属性：单例占位符，第一次 new 时填充

    # 【什么】单例构造
    # 【作用】保证全进程只有一个 MilvusVectorStore 实例
    # 【为什么】多实例 = 多份连接池 + 多份已加载集合缓存，会浪费连接和内存
    def __new__(cls):
        if cls._instance is None:                           # 第一次实例化
            cls._instance = super().__new__(cls)            # 调用父类 new 创建真实实例
        return cls._instance                                # 已存在则复用

    # 【什么】初始化
    # 【作用】读取配置、确定单/多集合参数、建连接池
    # 【为什么】__init__ 每次拿到实例都会执行，但单例保证对已有实例来说 __init__ 是幂等的——
    #   属性值由配置快照决定，配置没变的话再执行一次也不会破坏状态
    def __init__(self):
        s = get_settings()                                  # 取全局配置

        self._default_collection_name = s.milvus_collection  # 默认集合名：单集合模式就是它了

        # 兼容旧代码/外部直接读 _collection_name 的写法
        self._collection_name = s.milvus_collection         # 别名：老代码有人直接访问 _collection_name

        self._multi = bool(s.milvus_multi_collection)       # 是否多集合模式：bool() 防配置传字符串/整型
        self._prefix = s.milvus_collection_prefix           # 多集合前缀：如 "kb_"，拼出 "kb_finance"
        self._default_kb = s.milvus_default_kb              # 默认知识库编码：没指定 kb_code 时落这里

        self._dim = s.embedding_dimension                   # embedding 维度：必须与模型输出一致（本项目 1024）

        self._top_k = s.rag_top_k                           # 默认召回条数：检索时没传 top_k 就用这个

        self._threshold = s.similarity_threshold            # 相似度阈值：低于它的结果被过滤（防召回无关内容）

        # 集合名 → Collection 对象（多集合模式下懒加载，用到哪库才 load 哪库，
        # 避免启动时把所有知识库一次性灌进内存）
        self._collections: dict = {}                        # 缓存已加载的 Collection 对象：键 = 集合名

        self._connect()                                     # 立即建立 Milvus 连接并确保默认集合就绪

    # 【什么】Milvus 连接初始化
    # 【作用】建立连接、缓存 API 类、确保默认集合存在
    # 【为什么】连接断掉会触发 Milvus 的自动重连，所以这里只做一次性初始化；
    #   把 pymilvus 各类挂到 self 上是避免每个方法都重新 import，也便于 mock
    def _connect(self):
        """连接 Milvus 并确保默认集合存在。"""
        try:
            from pymilvus import (                           # 函数内 import：不把 pymilvus 绑在模块顶部，没装也能 import 本文件
                connections, utility, Collection,
                CollectionSchema, FieldSchema, DataType,
            )
            s = get_settings()                              # 再取一次配置（可能 connect 在测试里被独立调用）

            connections.connect(                             # 建立 Milvus gRPC 连接
                alias="default",                             # 连接别名：后续所有操作都用这个别名
                host=s.milvus_host,                          # 主机地址
                port=str(s.milvus_port),                     # 端口转字符串：pymilvus 期望字符串
            )

            self._connections = connections                   # 缓存 connections 模块
            self._utility = utility                           # 缓存 utility 模块（has_collection / drop_collection 等）
            self._Collection = Collection                     # 缓存 Collection 类
            self._DataType = DataType                         # 缓存 DataType 枚举
            self._FieldSchema = FieldSchema                   # 缓存 FieldSchema 类
            self._CollectionSchema = CollectionSchema         # 缓存 CollectionSchema 类
            # 默认集合始终就绪，保证单集合模式与旧行为一致
            self._collection = self._ensure_collection(self._default_collection_name)  # 确保默认集合存在并加载
            _logger.info(                                    # 连接成功日志
                f"Milvus连接成功: {s.milvus_host}:{s.milvus_port} | "  # 地址
                f"模式={'多集合' if self._multi else '单集合'}"         # 当前模式
            )
        except Exception as e:                              # 连接失败
            _logger.error(f"Milvus连接失败: {e}")            # 记错误日志
            raise RAGException("Milvus连接失败", str(e))     # 包装成 RAGException：上层统一处理

    # ------------------------------------------------------------------
    # 集合路由：kb_code ↔ 集合名
    # ------------------------------------------------------------------
    # 【什么】kb_code → 集合名
    # 【作用】根据知识库编码算出要操作的 Milvus 集合名
    # 【为什么】单集合模式所有数据在同一个物理集合里，kb_code 只作为标量字段存储；
    #   多集合模式每个库是独立物理集合，由前缀+kb_code 拼出名字
    def collection_name_for(self, kb_code: Optional[str]) -> str:
        """
        知识库编码 → Milvus 集合名。

        单集合模式一律返回默认集合（kb_code 只作为标量字段写入）；
        多集合模式返回 前缀 + kb_code，如 "finance" → "kb_finance"。
        """
        if not self._multi:                                 # 单集合模式
            return self._default_collection_name            # 无视 kb_code，全落一个集合
        return f"{self._prefix}{kb_code or self._default_kb}"  # 多集合模式：拼前缀 + 库码；空则用默认库

    # 【什么】集合名 → kb_code
    # 【作用】collection_name_for 的逆运算
    # 【为什么】检索结果要返回"来自哪个知识库"：多集合模式靠集合名反推，单集合模式靠标量字段
    def kb_of_collection(self, collection_name: str) -> str:
        """集合名 → 知识库编码（collection_name_for 的逆运算）。"""
        if not self._multi:                                 # 单集合模式
            return self._default_kb                         # 无法从集合名区分，只能返回默认库编码
        return collection_name[len(self._prefix):] if collection_name.startswith(self._prefix) else collection_name  # 多集合模式：去掉前缀即库码；防意外返回原集合名

    # 【什么】集合存在性保障
    # 【作用】若集合不存在则建表 + 建索引 + load，最后返回已就绪的 Collection 对象
    # 【为什么】Milvus 写入前必须确保集合存在；检索前必须 load 进内存——
    #   把"存在性检查 → 建表 → 建索引 → load"封装成原子操作，调用方拿到就能用
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
        if not self._utility.has_collection(name):          # 集合还不存在
            fields = [                                       # 字段清单：定义集合 schema
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
                field_name="embedding",                       # 给 embedding 列建索引
                index_params={                                # 索引参数
                    "index_type": "IVF_FLAT",                  # 倒排文件 + 平铺：精度高、内存中数据不经量化压缩
                    "metric_type": "COSINE",                   # 余弦距离：与向量模型的训练目标一致
                    "params": {"nlist": 128},                  # 128 个聚类桶：平衡检索速度与召回率
                },
            )
            _logger.info(f"Milvus集合创建: {name}")            # 建集合日志
        collection = self._Collection(name)                        # 集合此刻必然存在（刚建或原本有），取句柄
        self._collections[name] = collection                       # 先进缓存：下次 _get_collection 直接命中
        collection.load()                                          # load 进内存才能被检索；不 load 查询会报错
        _logger.info(
            f"Milvus集合就绪: {name} | 知识库隔离字段={self._has_field(name, 'kb_code')}"  # 顺带报告老集合有没有权限字段
        )
        return collection                                          # 返回已 load 的 Collection 对象

    # 【什么】带缓存取集合
    # 【作用】返回已加载的 Collection，首次访问时自动创建并 load
    # 【为什么】懒加载：避免启动时全量 load 所有知识库，只在检索/写入用到时才加载
    def _get_collection(self, name: str):
        """取集合（带缓存，首次访问才创建并 load）。"""
        collection = self._collections.get(name)            # 先查缓存
        if collection is None:                              # 没缓存过
            collection = self._ensure_collection(name)      # 建表 + load + 缓存一条龙
        return collection                                   # 返回句柄

    # 【什么】字段存在性检查
    # 【作用】判断集合是否有某个标量字段（比如 kb_code）
    # 【为什么】老集合升级前可能没有 kb_code 字段，检索时若强制按 kb_code 过滤会报错；
    #   先检查再决定要不要拼 kb_code 条件
    def _has_field(self, name: str, field: str) -> bool:
        """判断集合有没有某个标量字段（老集合可能没有 kb_code，需要兼容）。"""
        try:
            return any(f.name == field for f in self._get_collection(name).schema.fields)  # 遍历 schema.fields 比较字段名
        except Exception:                                   # 集合本身不存在等异常
            return False                                    # 保守返回 False：没有该字段

    # 【什么】kb_codes → 集合名列表
    # 【作用】把知识库编码列表翻译成要访问的 Milvus 集合名列表
    # 【为什么】多集合模式下跨库检索要搜多个集合；单集合模式始终一个集合
    def _target_collections(self, kb_codes: Optional[list]) -> list:
        """把 kb_codes 转成要访问的集合名列表；为空时退回默认集合。"""
        if not kb_codes:                                    # 没指定知识库
            return [self.collection_name_for(self._default_kb)] if self._multi else [self._default_collection_name]  # 多集合返回默认库集合，单集合返回默认集合
        return [self.collection_name_for(c) for c in kb_codes]  # 逐个翻译 kb_code → 集合名

    # 【什么】列出已有知识库
    # 【作用】返回当前已存在（有物理集合）的知识库编码列表
    # 【为什么】管理后台需要展示"有哪些知识库"；多集合模式下每个库对应一个物理集合
    def list_knowledge_bases(self) -> list:
        """列出已存在的知识库编码。

        多集合模式：扫描 Milvus 里所有带前缀的集合名反推（如 kb_finance → finance）。
        单集合模式：只有一个逻辑库，返回默认库编码。
        """
        if not self._multi:                                 # 单集合模式
            return [self._default_kb]                       # 物理上只有一个集合
        try:
            names = self._utility.list_collections() or []  # 列举 Milvus 里所有集合名；None 时兜底空列表
        except Exception as e:                              # 连接中断等
            _logger.warning(f"列举Milvus集合失败: {e}")     # 记警告
            return [self._default_kb]                       # 保底返回默认库
        return [n[len(self._prefix):] for n in names if n.startswith(self._prefix)]  # 多集合模式：过滤出带前缀的集合，去前缀反推 kb_code

    # 【什么】整库下线
    # 【作用】删除某个知识库对应的所有数据
    # 【为什么】多集合模式物理隔离，可以整库 drop；单集合模式只能按 doc_id 逐删
    @handle_exception(default=False, log_msg="Milvus知识库下线异常")  # 装饰器：异常返回 False 不抛错
    def drop_kb(self, kb_code: str) -> bool:
        """整库下线：直接 drop 掉整个物理集合（多集合模式独有的能力）。

        单集合模式下没法这么做（会把别人的库一起删了），只能按 doc_id 下线文档。
        """
        if not self._multi:                                 # 单集合模式
            _logger.warning("单集合模式不支持整库 drop，请改用 remove_by_doc_id 按文档下线")  # 警告 + 给出替代方案
            return False                                    # 操作被拒绝
        name = self.collection_name_for(kb_code)            # 算出该库对应的集合名
        if self._utility.has_collection(name):              # 集合存在才删
            self._utility.drop_collection(name)             # 物理删除集合（连同数据、索引一起清掉）
            self._collections.pop(name, None)               # 清掉本地缓存中的旧句柄（否则下次用会指向已删集合）
            _logger.info(f"知识库已整库删除: kb={kb_code} | 集合={name}")  # 下线日志
        return True                                         # 返回成功（集合不存在也算成功，幂等）

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    # 【什么】分片批量入库
    # 【作用】把 TextChunk 列表向量化后写入 Milvus
    # 【为什么】一批 chunks 里可能混有多个 kb_code，自动按目标集合分组写入——
    #   上层只管扔分片，不用管单集合/多集合的路由
    def insert_chunks(self, chunks: List[TextChunk], kb_code: Optional[str] = None) -> int:
        """
        将文本分片向量化后入库。

        Args:
            chunks: 文本分片
            kb_code: 目标知识库。为空时取 chunk.metadata["kb_code"]，再没有就用默认库。
                     一批 chunks 里 kb_code 不同也没关系，会按目标集合自动分组写入。
        """
        if not chunks:                                      # 空列表短路
            return 0                                        # 直接返回 0
        emb_service = get_embedding_service()               # 取向量化服务实例（通常也是单例）
        vectors = emb_service.embed_batch([c.content for c in chunks])  # 批量向量化：把所有分片正文一次性过模型（比逐条快得多）
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 统一入库时间戳：所有分片同一次写入共享同一个时间

        # 按目标集合分组：多集合模式下不同 kb_code 落进不同物理集合
        groups: dict = {}                                   # 字典：{集合名: [(i, code, chunk), ...]}
        for i, c in enumerate(chunks):                      # 逐片枚举
            code = kb_code or c.metadata.get("kb_code") or self._default_kb  # 目标库编码：参数 > 分片元数据 > 默认库
            groups.setdefault(self.collection_name_for(code), []).append((i, code, c))  # 按集合名分组

        total = 0                                           # 累计写入条数
        for name, items in groups.items():                  # 逐集合写入
            collection = self._get_collection(name)         # 获取/创建集合并 load
            data = [                                         # 按列组织数据：Milvus insert 要求"按字段给列表"
                [c.metadata.get("chunk_id", f"chunk_{i}") for i, _, c in items],  # chunk_id 列：没显式指定则自动生成
                [c.metadata.get("doc_id", "") for _, _, c in items],     # doc_id 列
                [c.content for _, _, c in items],             # content 列
                [c.metadata.get("source", "") for _, _, c in items],     # source 列
                [c.metadata.get("file_name", "") for _, _, c in items],  # file_name 列
                [now] * len(items),                           # updated_at 列：统一时间
            ]
            if self._has_field(name, "kb_code"):             # 集合有 kb_code 字段（新集合/老集合升级后才有）
                data.append([code for _, code, _ in items])   # kb_code 列：该分片归属哪个知识库
                data.append([c.metadata.get("dept_code", "") for _, _, c in items])  # dept_code 列
            data.append([vectors[i] for i, _, _ in items])   # embedding 列：按 groups 里的原始索引 i 取对应向量
            collection.insert(data)                          # 执行批量插入
            collection.flush()                               # 同步刷盘：确保刚写入的数据可被检索（默认写进 growing segment，flush 后转成 sealed segment）
            total += len(items)                              # 累加写入数
            _logger.info(f"向量入库完成: 集合={name} | {len(items)} 条分片")  # 单集合写入日志
        return total                                         # 返回总写入条数

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    # 【什么】向量相似度召回
    # 【作用】把 query 向量化后，在目标知识库的集合里做 ANN 检索，返回相关文档列表
    # 【为什么】@handle_exception(default=None, reraise=False)：检索异常时返回 None 而不是抛错——
    #   上层 RAGChain.retrieve 收到 None 会走空结果处理（边界1降级），保证问答主链路不崩
    @handle_exception(default=None, log_msg="Milvus向量召回异常", reraise=False)  # 装饰器：异常记日志并返回 None
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
        emb_service = get_embedding_service()               # 取向量化服务
        query_vec = emb_service.embed(text=query)           # 把用户 query 转成向量（单条 embed）
        k = top_k or self._top_k                            # 召回条数：参数 > 配置默认值

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
        #     'updated_at > "2026-01-01"（无 kb_code）                 → [] | 原样返回
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
        targets = self._target_collections(codes)           # 转成要搜的集合名列表

        hits = []                                           # 收集所有集合的命中结果
        for name in targets:                                # 逐集合检索（单集合循环 1 次，多集合循环 N 次）
            try:
                hits.extend(self._search_in(name, query_vec, k, inner_expr))  # 在单个集合里搜，结果追加到 hits
            except Exception as e:                          # 某个集合挂了（没建索引 / 连接闪断等）
                # 某一个库挂了不能让整个检索失败（比如该集合还没建索引）
                _logger.warning(f"集合 {name} 检索失败，已跳过: {e}")  # 记警告：跳过坏集合，继续搜其他集合

        hits.sort(key=lambda h: h["score"], reverse=True)   # 全局按相似度降序：多集合合并后必须重排
        hits = hits[:k]                                     # 截断到 top_k
        _logger.info(                                      # 检索结果日志
            f"向量召回: query='{query[:30]}...' | 命中={len(hits)}/{k} | "  # query 截断防日志爆炸
            f"集合={[t for t in targets]} | 知识库过滤={'有' if (expr or kb_codes) else '无'}"  # 报告实际检索的集合和是否有过滤
        )
        return hits                                         # 返回整理后的命中列表

    # 【什么】单集合内检索
    # 【作用】在某个具体集合上执行一次 ANN，整理出结构化结果
    # 【为什么】封装一层方便处理"集合缺 kb_code 字段"等兼容场景，也统一结果字段名
    def _search_in(self, name: str, query_vec: list, k: int, expr: Optional[str]) -> list:
        """在单个集合里做一次 ANN 检索并整理结果。"""
        collection = self._get_collection(name)             # 获取集合并 load
        base_fields = ["chunk_id", "doc_id", "content", "source", "file_name", "updated_at"]  # 基础返回字段
        extra_fields = ["kb_code", "dept_code"] if self._has_field(name, "kb_code") else []  # 老集合可能没有 kb_code/dept_code
        try:
            results = self._search_once(collection, query_vec, k, expr, base_fields + extra_fields)  # 执行 ANN 检索
        except Exception as e:                              # 第一次检索失败
            if not expr:                                    # 没过滤条件还失败 = 不是过滤字段缺失的问题
                raise                                       # 直接抛：可能是连接断了或索引没建好
            _logger.warning(f"集合 {name} 过滤检索失败，回退无过滤（疑似集合缺 kb_code 字段）: {e}")  # 怀疑是过滤字段缺失
            results = self._search_once(collection, query_vec, k, None, base_fields)  # 回退无过滤重搜（仅返回基础字段）

        kb = self.kb_of_collection(name)                    # 该集合对应的知识库编码（多集合模式从名字反推，单集合模式默认）
        hits = []                                           # 收集命中
        # pymilvus 的 search 返回类型被标注成 Future/Coroutine 联合，实际是同步结果列表
        for hit in results[0]:   # type: ignore              # results[0] = 第 1 批（也是唯一一批，因为查询向量批次=1）
            score = hit.score                               # 相似度分数（余弦距离：范围 [-1, 1]，越接近 1 越相关）
            if score < self._threshold:                     # 低于阈值
                _logger.debug(f"低相关结果过滤: score={score:.4f} < {self._threshold}")  # 调试用 debug 日志
                continue                                    # 跳过这条
            hits.append({                                     # 组装结构化结果
                "content": hit.entity.get("content", ""),    # 分片正文
                "source": hit.entity.get("source", ""),      # 来源路径
                "file_name": hit.entity.get("file_name", ""),# 文件名
                "doc_id": hit.entity.get("doc_id", ""),      # 文档 id
                "updated_at": hit.entity.get("updated_at", ""),  # 更新时间
                "score": round(score, 4),                    # 分数保留 4 位小数
                # 单集合模式下 kb_code 来自字段值；多集合模式下来自集合名
                "kb_code": hit.entity.get("kb_code", kb) if "kb_code" in extra_fields else kb,
                "collection": name,                          # 来自哪个集合（排查用）
            })
        return hits                                         # 返回单集合命中列表

    # 【什么】关键词降级召回
    # 【作用】向量召回无结果时，用 Milvus 的标量 like 做关键词粗召回
    # 【为什么】边界1：embedding 对生僻词/缩写/新词可能完全失配，关键词匹配是最后的保命通道；
    #   命中结果的 score 统一置为 threshold：避免"降级结果分数为 0 又被阈值过滤掉"的二次丢失
    @handle_exception(default=[], log_msg="Milvus关键词降级检索异常")  # 装饰器：异常时返回空列表
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
            w for w in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]{2,}", query)  # 正则提取关键词：中文至少 2 字，英文/数字至少 2 个字符
        ]
        if not keywords:                                    # 没提取到任何可用关键词
            return []                                       # 无法降级，返回空列表
        keywords.sort(key=len, reverse=True)                # 按长度降序：长词更具体、召回更精准
        keywords = keywords[:3]                             # 只取最长的前 3 个：防止短词泛召回太多无关结果

        conds = " or ".join(f'content like "%{kw}%"' for kw in keywords)  # Milvus like 表达式：content 字段包含任一关键词

        codes = list(kb_codes) if kb_codes else self._parse_kb_codes(expr)  # 知识库列表：同 search 的逻辑
        inner_expr = self._strip_kb_expr(expr) if self._multi else expr    # 库内过滤条件：同 search 的逻辑
        targets = self._target_collections(codes)           # 要搜的集合名列表

        limit = top_k or self._top_k                        # 召回条数
        hits = []                                           # 收集命中
        for name in targets:                                # 逐集合关键词检索
            final_expr = f"({conds})" + (f" and {inner_expr}" if inner_expr else "")  # 拼接关键词条件与权限条件
            try:
                rows = self._get_collection(name).query(     # Milvus query（非 ANN，纯标量过滤）
                    expr=final_expr,                         # 过滤表达式
                    output_fields=["chunk_id", "doc_id", "content", "source", "file_name", "updated_at"],  # 返回字段
                    limit=limit,                             # 上限
                )
            except Exception as e:                          # 单个集合查询失败
                _logger.warning(f"集合 {name} 关键词检索失败，已跳过: {e}")  # 记警告后跳过
                continue                                    # 继续下一个集合
            kb = self.kb_of_collection(name)                # 该集合的知识库编码
            hits.extend({                                    # 逐行组装结果
                "content": r.get("content", ""),
                "source": r.get("source", ""),
                "file_name": r.get("file_name", ""),
                "doc_id": r.get("doc_id", ""),
                "updated_at": r.get("updated_at", ""),
                "score": self._threshold,                    # 关键词召回没有向量分，统一塞 threshold，让下游 reranker 有机会重排
                "kb_code": kb,
                "collection": name,
                "recall_type": "keyword",                    # 标记来源：便于监控区分向量召回 vs 关键词降级
            } for r in rows)   # type: ignore                # rows 实际类型与标注不符，忽略类型检查

        hits.sort(key=lambda h: h["score"], reverse=True)   # 按 score 降序（虽然都是 threshold，但保留排序接口一致性）
        hits = hits[:limit]                                 # 截断
        _logger.info(f"关键词降级召回: 关键词={keywords} | 命中={len(hits)}")  # 降级通道结果日志
        return hits                                         # 返回关键词召回结果

    # 【什么】从 expr 反解析 kb_code 列表
    # 【作用】把 'kb_code in ["a","b"]' 里的 a,b 提取成 Python 列表
    # 【为什么】多集合模式下 kb_code 条件被翻译成"搜哪几个集合"，所以必须先提取出列表
    @staticmethod                                           # 静态方法：纯工具函数，不依赖实例状态
    def _parse_kb_codes(expr: Optional[str]) -> list:
        """从 'kb_code in ["public","finance"]' 里提取出 ['public','finance']。

        这样 RBAC 生成的表达式不用改就能驱动多集合路由：
        表达式里的 kb_code 条件在多集合模式下被翻译成"搜哪几个集合"。
        """
        if not expr:                                        # 空表达式
            return []                                       # 没有限定知识库
        m = re.search(r'kb_code\s+in\s*\[(.*?)\]', expr)   # 正则匹配 kb_code in [...] 片段；非贪婪匹配括号内内容
        if not m:                                           # 没匹配到
            return []                                       # 表达式里没有 kb_code in 条件
        return [x.strip().strip('"\'') for x in m.group(1).split(",") if x.strip()]  # 按逗号拆分，去空白去引号，过滤空值

    # 【什么】剥离 kb_code 条件
    # 【作用】把 expr 里的 kb_code in [...] 片段去掉，留下剩余过滤条件
    # 【为什么】多集合模式下 kb_code 体现在集合名上，集合内再过滤 kb_code 会报错（老集合缺字段）
    @staticmethod                                           # 静态方法：纯工具函数
    def _strip_kb_expr(expr: Optional[str]) -> Optional[str]:
        """去掉表达式里的 kb_code in [...] 片段，留下集合内的其他过滤条件。

        'kb_code in ["a","b"] and dept_code == "D_FIN"' → 'dept_code == "D_FIN"'
        """
        if not expr:                                        # 空表达式
            return None                                     # 没有剩余条件
        m = re.search(r'kb_code\s+in\s*\[.*?\]', expr)     # 匹配 kb_code in [...] 片段
        if not m:                                           # 没匹配到
            return expr                                     # 原样返回（没有 kb_code 可剥）
        rest = expr.replace(m.group(0), "")                 # 去掉匹配到的 kb_code 片段
        rest = re.sub(r'^\s*(and|or)\s*', '', rest, flags=re.I).strip()  # 去掉开头残留的 and/or（不区分大小写）
        rest = re.sub(r'\s*(and|or)\s*$', '', rest, flags=re.I).strip()  # 去掉末尾残留的 and/or
        return rest or None                                 # 剥完空了返回 None（集合内不过滤）；有残留返回残余条件

    # 【什么】执行单次 ANN 检索
    # 【作用】调用 pymilvus Collection.search，把参数包正确
    # 【为什么】expr=None 与"不传"在部分 Milvus 版本不等价，所以 expr 只在非空时才塞进 kwargs
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
        if expr:                                            # 有过滤条件
            kwargs["expr"] = expr                           # 动态加入 expr 键
        return collection.search(**kwargs)                  # 执行检索并返回原始结果

    # ------------------------------------------------------------------
    # 删除与统计
    # ------------------------------------------------------------------
    # 【什么】按文档 ID 下线
    # 【作用】删除指定 doc_id 的所有分片
    # 【为什么】边界5：知识库过期管理——新文档导入后旧版同名文档需要清理；
    #   多集合模式下没指定 kb_code 时遍历所有知识库集合逐个删
    @handle_exception(default=False, log_msg="Milvus文档下线异常")  # 装饰器：异常返回 False
    def remove_by_doc_id(self, doc_id: str, kb_code: Optional[str] = None) -> bool:
        """按文档ID下线过期文档（边界5：知识库过期管理）。

        kb_code 为空且处于多集合模式时，会在所有知识库集合里逐个删。
        """
        expr = f'doc_id == "{doc_id}"'                      # Milvus 删除表达式：精确匹配 doc_id
        if not self._multi:                                 # 单集合模式
            names = [self._default_collection_name]         # 只搜默认集合
        elif kb_code:                                       # 多集合模式且指定了库
            names = [self.collection_name_for(kb_code)]     # 只删该库的集合
        else:                                               # 多集合模式且没指定库
            names = [self.collection_name_for(k) for k in self.list_knowledge_bases()]  # 遍历所有已有知识库集合

        for name in names:                                  # 逐集合删除
            collection = self._get_collection(name)         # 获取集合并 load
            collection.delete(expr)                          # 按 doc_id 精确删除
            collection.flush()                               # 同步刷盘：确保删除立即生效（否则查询可能还能看到旧数据）
        _logger.info(f"文档下线完成: doc_id={doc_id} | 涉及集合={names}")  # 下线结果日志
        return True                                         # 返回成功

    # 【什么】向量总数统计
    # 【作用】返回所有目标知识库的分片总数
    # 【为什么】调用 count_by_kb 再求和，逻辑复用；空 kb_codes = 全部知识库
    @handle_exception(default=0, log_msg="Milvus统计异常")  # 装饰器：异常返回 0
    def count(self, kb_codes: Optional[list] = None) -> int:
        """统计向量总数。kb_codes 为空 = 统计全部知识库（单集合模式即整个集合）。"""
        return sum(self.count_by_kb(kb_codes).values())     # 分库统计后求和

    # 【什么】分库统计
    # 【作用】返回 {kb_code: 分片数} 的字典
    # 【为什么】flush 是关键：num_entities 只统计已落盘的段，刚写入未 flush 的数据会被漏掉；
    #   但 flush 是同步阻塞操作，高频调用会拖慢写入吞吐，所以统计接口才有意义，不要写在写入链路里
    @handle_exception(default={}, log_msg="Milvus分库统计异常")  # 装饰器：异常返回空字典
    def count_by_kb(self, kb_codes: Optional[list] = None) -> dict:
        """按知识库分别统计：{"public": 120, "finance": 45}

        多集合模式下每个库是独立集合，天然能分开数；
        单集合模式下只有一个逻辑库，返回 {默认库: 总数}。
        """
        names = self._target_collections(kb_codes)          # 要统计的集合名列表
        out = {}                                            # 结果字典
        for name in names:                                  # 逐集合统计
            try:
                collection = self._get_collection(name)     # 获取集合并 load
                # 把内存缓冲区（growing segment）里刚写入的向量刷成已持久化段。
                # 不 flush 的话 num_entities 只数已落盘的数据，刚入库的文档会被漏掉 → 计数偏小。
                # 代价：flush 是同步阻塞调用，会等落盘完成，高频调用会拖慢写入吞吐。
                collection.flush()                           # 同步刷盘：保证计数准确
                out[self.kb_of_collection(name)] = collection.num_entities  # 取该集合实体数（= 行数）
            except Exception as e:                          # 某个集合统计失败
                _logger.warning(f"集合 {name} 统计失败: {e}")  # 记警告
                out[self.kb_of_collection(name)] = 0        # 该集合计数填 0，不影响其他集合
        return out                                          # 返回分库统计结果
