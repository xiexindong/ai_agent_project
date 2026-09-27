"""端到端演示：一个问句如何一步步被检索出来（真实 Milvus + 真实 embedding）。

运行方式：python _demo_rag.py
会在本地 Milvus 建 3 个临时集合 demo_kb_public / demo_kb_finance / demo_kb_hr，
跑完后自动 drop 清理。
"""
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 关键：必须在 import config 之前设置，才能切到多集合模式演示"每个库一个物理集合"
os.environ["MILVUS_MULTI_COLLECTION"] = "true"
os.environ["MILVUS_COLLECTION_PREFIX"] = "demo_kb_"

from modules.rag.vector_store import MilvusVectorStore          # noqa: E402
from modules.rag.text_splitter import TextChunk                  # noqa: E402
from modules.rag.reranker import BGEReranker                     # noqa: E402
from modules.enterprise.models import Principal                  # noqa: E402
from modules.enterprise.rbac_service import RBACService          # noqa: E402
from modules.memory.embedding_service import get_embedding_service  # noqa: E402

QUERY = "差旅住宿费报销标准是多少"

DOCS = {
    "public": [
        "公司通用差旅规定：出差需提前在 OA 提交申请，经部门经理审批后方可订票。",
        "报销时限：差旅结束后 30 个工作日内提交报销单，逾期不予受理。",
    ],
    "finance": [
        "差旅住宿费报销标准：一线城市每晚不超过 600 元，二线城市 450 元，需提供增值税专用发票。",
        "市内交通费报销：凭票实报实销，单日上限 150 元。",
    ],
    "hr": [
        "年假规则：工龄满 1 年可享 5 天年假，满 10 年 10 天。",
        "调休申请需在加班后 90 天内使用完毕。",
    ],
}


def line(t):
    print("\n" + "=" * 66)
    print(t)
    print("=" * 66)


def main():
    store = MilvusVectorStore()

    line("第 0 步  环境")
    print("模式:", "多集合" if store._multi else "单集合", "| 前缀:", store._prefix)
    for kb in DOCS:
        print(f"  {kb:8s} -> 集合 {store.collection_name_for(kb)}")

    line("第 1 步  入库（3 个知识库，各自落到独立集合）")
    for kb, texts in DOCS.items():
        chunks = [
            TextChunk(
                content=t,
                metadata={"kb_code": kb, "doc_id": f"{kb}-{i}", "chunk_id": f"{kb}-{i}-0",
                          "source": f"{kb}.md", "file_name": f"{kb}.md"},
                chunk_index=i,
                total_chunks=len(texts),
            )
            for i, t in enumerate(texts)
        ]
        n = store.insert_chunks(chunks, kb_code=kb)
        print(f"  集合 {store.collection_name_for(kb)}: 写入 {n} 条")
    print("  分库统计:", store.count_by_kb(["public", "finance", "hr"]))

    line("第 2 步  身份 -> 可见知识库")
    principal = Principal(user_code="U007", dept_code="D_FIN", kb_codes=["public", "finance"])
    expr = RBACService().kb_expr(principal)
    print("  Principal.kb_codes =", principal.kb_codes)
    print("  kb_expr() =", expr)
    print("  注意：hr 库不在列表里 -> 本次检索根本不会访问 demo_kb_hr")

    line("第 3 步  表达式 -> 要搜的集合")
    print("  _parse_kb_codes(expr) =", MilvusVectorStore._parse_kb_codes(expr))
    print("  _target_collections() =", store._target_collections(["public", "finance"]))

    line("第 4 步  每个集合各搜一次（问句不挑库，逐库都搜）")
    qv = get_embedding_service().embed(QUERY)
    print("  问句向量维度:", len(qv))
    for kb in ["public", "finance"]:
        name = store.collection_name_for(kb)
        rows = store._search_in(name, qv, 5, None)
        print(f"  集合 {name}: {len(rows)} 条")
        for r in rows:
            print(f"      score={r['score']:.4f}  {r['content'][:28]}")

    line("第 5 步  合并 + 全局排序 + 阈值过滤（threshold={}）".format(store._threshold))
    hits = store.search(QUERY, top_k=5, expr=expr)
    for h in hits or []:
        print(f"  score={h['score']:.4f}  kb={h['kb_code']:8s} 集合={h['collection']:16s} {h['content'][:28]}")

    line("第 6 步  精排收口（FlagEmbedding 未装 -> 走降级加权）")
    reranker = BGEReranker()
    print("  模型:", reranker._model)
    top = reranker.rerank(QUERY, list(hits or []), top_k=3)
    for d in top:
        print(f"  rerank_score={d['rerank_score']:.4f}  kb={d.get('kb_code')}  {d['content'][:30]}")

    line("第 7 步  清理临时集合")
    for kb in DOCS:
        store.drop_kb(kb)
    print("  已 drop:", [store.collection_name_for(k) for k in DOCS])


if __name__ == "__main__":
    main()
