"""
Step 3: 构建 FAQ + 商品的 FAISS 索引
- 读 Step 1 数据 + Step 2 文本 → Step 3 embedder 编码 → FAISS 落盘
- 双索引：faq.index + products.index（Step 4 双 retriever 用）

用法：
    D:/Anaconda3/envs/graph/python.exe src/rag/build_index.py
"""
import time
from pathlib import Path

import faiss
import numpy as np

import json                            
from build_corpus import build_faq_corpus, build_products_corpus
from load_data import load_faq, load_products
from embedder import embed_texts, VECTOR_DIM 

# === 路径常量 ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VECTOR_DIR = PROJECT_ROOT / "data" / "vector_store"
VECTOR_DIR.mkdir(parents=True, exist_ok=True)  # 自动建目录

def _save_index(texts: list[str], metadatas: list, name: str) -> int:
    """通用索引构建 + 落盘

    1. 用 embedder 编码 texts
    2. 创建 IndexFlatIP（cosine via inner product，因为向量已归一化）
    3. 落盘到 data/vector_store/{name}.index
    4. 落盘元数据到 data/vector_store/{name}.meta.json

    Args:
        texts: 可检索文本列表
        metadatas: 与 texts 一一对应的原始 dict 列表
        name: 索引名（faq / products）

    Returns:
        写入的文本条数
    """
    print(f"\n--- [{name}] 编码 {len(texts)} 条 ---")
    t0 = time.perf_counter()
    vecs = embed_texts(texts)
    print(f"  编码耗时: {time.perf_counter() - t0:.1f}s, shape={vecs.shape}")
    
    print(f"  构建 IndexFlatIP ...")
    index = faiss.IndexFlatIP(VECTOR_DIM)
    index.add(np.ascontiguousarray(vecs))  # FAISS 要 C-contiguous

    # 落盘
    index_path = VECTOR_DIR / f"{name}.index"
    meta_path = VECTOR_DIR / f"{name}.meta.json"
    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)

    size_mb = index_path.stat().st_size / 1024**2
    print(f"  索引: {index_path.name} ({size_mb:.1f} MB)")
    print(f"  元数据: {meta_path.name} ({len(metadatas)} 条)")
    return len(texts)

def build_faq_index() -> int:
    """从 load_faq() + build_faq_corpus() 构建 FAQ 索引"""
    faq_list = load_faq()
    texts, metas = build_faq_corpus(faq_list)
    return _save_index(texts, metas, "faq")

def build_products_index() -> int:
    """从 load_products() + build_products_corpus() 构建商品索引"""
    products_list = load_products()
    texts, metas = build_products_corpus(products_list)
    return _save_index(texts, metas, "products")

if __name__ == "__main__":
    print("=" * 50)
    print("Step 3: 构建双 FAISS 索引")
    print("=" * 50)

    n1 = build_faq_index()
    print(f"\nFAQ 完成: {n1} 条")

    n2 = build_products_index()
    print(f"\n商品完成: {n2} 条")

    print("\n" + "=" * 50)
    print(f"全部完成。下一步：python test/test_rag.py 测试检索")
    print("=" * 50)