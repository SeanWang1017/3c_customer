"""
LangChain Tool 集合 —— 给 Agent 用
- query_order: 查订单状态（mock JSON）
- search_faq: 查售后 FAQ（FAISS 向量库）—— 走子进程，防 segfault
- search_product: 查商品（FAISS 向量库）—— 走子进程，防 segfault

参照 ORDERAGENT 的 agent/langchain_assistant.py 多个 @tool 模式

RAG 走子进程的原因：见 memory/openpyxl-sentence-transformers-segfault.md
"""
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from langchain.tools import tool

from src.tools.order_query import query_order as _query_order_impl

# ============ 子进程配置（避免主进程 import faiss/sentence_transformers）===========
_RAG_SCRIPT = PROJECT_ROOT / "scripts" / "rag_query.py"
_RAG_PYTHON = "D:/Anaconda3/envs/graph/python.exe"
# 强制子进程用 UTF-8（Windows 默认 GBK 会导致中文解码失败）
_RAG_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def _rag_subprocess(index_name: str, query: str, top_k: int = 3) -> list[dict]:
    """调 scripts/rag_query.py 子进程，返回 hits 列表。"""
    try:
        result = subprocess.run(
            [_RAG_PYTHON, str(_RAG_SCRIPT), index_name, query, str(top_k)],
            stdout=subprocess.PIPE,         # 捕获 stdout（JSON）
            stderr=subprocess.DEVNULL,      # 屏蔽 sentence_transformers 进度条
            text=True,
            encoding="utf-8",
            env=_RAG_ENV,
            timeout=120,
        )
        if result.returncode != 0:
            print(
                f"[rag] subprocess exit={result.returncode} index={index_name} "
                f"query={query[:30]!r}",
                file=sys.stderr,
            )
            return []
        return json.loads(result.stdout)
    except Exception as e:
        # 子进程崩溃 / 超时 / JSON 解析失败时留痕，避免 RAG 命中为空时分不清原因
        print(f"[rag] subprocess failed: {type(e).__name__}: {e}", file=sys.stderr)
        return []


# ============ 1. 订单工具 ============
@tool
def query_order(order_id: str) -> dict:
    """查询订单状态（Mock 数据）。

    Args:
        order_id: 任意订单号字符串，找不到时返回 found=False。

    Returns:
        dict: 含 found/status/logistics；订单不存在时 found=False, message 说明原因。
    """
    return _query_order_impl(order_id)


# ============ 2. FAQ 工具 ============
@tool
def search_faq(query: str) -> str:
    """在 FAQ 知识库中检索售后政策相关问题。

    适用于：用户问"激活能不能退"、"三包政策"、"保修多久"等售后问题。

    Args:
        query: 用户的问题

    Returns:
        str: 命中的 FAQ 条目（前 3 条），格式：编号 + 相似度 + 问题 + 答案
    """
    hits = _rag_subprocess("faq", query, top_k=3)
    if not hits:
        return "未找到相关 FAQ"
    lines = []
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        q_text = m.get("question", str(m)[:80])
        a_text = m.get("answer", "")
        lines.append(f"[{i}] (相似度 {h['score']:.2f}) Q: {q_text}\n    A: {a_text}")
    return "\n".join(lines)


# ============ 3. 商品工具 ============
@tool
def search_product(query: str) -> str:
    """在商品知识库中检索 3C 商品。

    适用于：用户问"推荐 X"、"Y 怎么样"、"Z 的参数"等商品问题。

    Args:
        query: 用户的问题

    Returns:
        str: 命中的商品信息（前 3 条），格式：编号 + 相似度 + 标题 + 价格
    """
    hits = _rag_subprocess("products", query, top_k=3)
    if not hits:
        return "未找到相关商品"
    lines = []
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        title = m.get("title") or m.get("product_name") or str(m)[:60]
        price = m.get("price", "?")
        lines.append(f"[{i}] (相似度 {h['score']:.2f}) {title} - ¥{price}")
    return "\n".join(lines)

