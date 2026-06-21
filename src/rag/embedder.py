"""
Step 3: 加载 BGE-small-zh-v1.5 + 文本向量化
"""
import importlib
from functools import lru_cache
from pathlib import Path

# === 路径与常量 ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EMBEDDING_MODEL_PATH = PROJECT_ROOT / "models" / "embeddings" / "bge-small-zh-v1.5"
VECTOR_DIM = 512  # bge-small-zh 输出维度

@lru_cache(maxsize=1)
def get_embedder():
    """单例加载 SentenceTransformer 模型（从本地路径）

    用 importlib.import_module 而非直接 from import，避免和 faiss 同时
    加载导致 segfault（详见 memory/openpyxl-sentence-transformers-segfault.md）
    """
    st_module = importlib.import_module("sentence_transformers")
    if not EMBEDDING_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"BGE 模型不存在: {EMBEDDING_MODEL_PATH}\n"
            f"请从 HF 下载到该目录，或检查 models/embeddings/ 路径"
        )
    return st_module.SentenceTransformer(str(EMBEDDING_MODEL_PATH))

def embed_texts(texts):
    """批量编码文本为向量

    Args:
        texts: 字符串列表

    Returns:
        numpy.ndarray of shape (len(texts), VECTOR_DIM), 已归一化
    """
    model = get_embedder()
    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    )

def embed_query(query: str):
    """编码单条 query，返回 (1, dim) 的 numpy 数组"""
    return embed_texts([query])
