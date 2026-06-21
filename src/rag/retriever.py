"""
Step 4: 双索引检索器
- 读 Step 3 落盘的 faq.index / products.index
- 根据 intent 路由到对应索引
- 返回 top_k 命中 + 原始 dict

用法（Step 5 / S5 Agent 会用）：
    from retriever import retrieve
    hits = retrieve(intent="policy_qa", query="7天无理由怎么算", top_k=3)
"""
import json
import sys
import importlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.embedder import embed_query

# 懒加载 faiss（避免和 sentence_transformers 同时 import 导致 segfault）
_faiss_module = None


def _get_faiss():
    global _faiss_module
    if _faiss_module is None:
        _faiss_module = importlib.import_module("faiss")
    return _faiss_module

# === 路径常量 ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VECTOR_DIR = PROJECT_ROOT / "data" / "vector_store"

# 意图 → 索引名 路由表
INTENT_TO_INDEX = {
    "policy_qa": "faq",
    "order_query": "products",
    "product_intro": "products",
    "product_recommend": "products",
    "ticket_transfer": "products",
}

class FaissRetriever:
    """单个 FAISS 索引的检索器"""

    def __init__(self, name: str):
        self.name = name
        index_path = VECTOR_DIR / f"{name}.index"
        meta_path = VECTOR_DIR / f"{name}.meta.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"索引不存在: {index_path}\n"
                f"请先跑 python src/rag/build_index.py"
            )
        self.index = _get_faiss().read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metas = json.load(f)
        print(f"[retriever:{name}] 加载完成, {self.index.ntotal} 条索引")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """检索 top_k 命中

        Returns:
            [{"score": float, "meta": dict}, ...]
        """
        q_vec = embed_query(query)  # (1, 512)
        scores, indices = self.index.search(q_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metas):
                continue
            results.append({
                "score": float(score),
                "meta": self.metas[idx],
            })
        return results

def retrieve(intent: str, query: str, top_k: int = 5) -> list[dict]:
    """根据意图路由到对应索引，返回 top_k 命中

    路由表：INTENT_TO_INDEX（模块顶部）

    Args:
        intent: 5 类意图之一
        query: 用户问题
        top_k: 返回前 k 个

    Returns:
        list of {"score": float, "meta": dict}
    """
    index_name = INTENT_TO_INDEX.get(intent)
    if index_name is None:
        raise ValueError(
            f"未知意图: {intent!r}, 应为 {list(INTENT_TO_INDEX.keys())}"
        )
    retriever = FaissRetriever(index_name)
    return retriever.retrieve(query, top_k)

if __name__ == "__main__":
    # 验收：4 个 query × 不同意图
    test_cases = [
        ("policy_qa", "7天无理由退货怎么算"),
        ("policy_qa", "保修期多久"),
        ("product_intro", "这款笔记本多大内存"),
        ("product_recommend", "推荐 5000 以内的游戏本"),
    ]
    for intent, query in test_cases:
        print(f"\n--- [{intent}] {query} ---")
        hits = retrieve(intent, query, top_k=3)
        for i, h in enumerate(hits, 1):
            m = h["meta"]
            # FAQ 显示 question, 商品显示 title
            if intent == "policy_qa":
                text = m.get("question", "?")[:60]
            else:
                text = m.get("title", "?")[:60]
            print(f"  [{i}] {h['score']:.3f}  {text}")
