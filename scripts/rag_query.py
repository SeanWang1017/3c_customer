"""
RAG 子进程入口（解决 faiss + sentence_transformers + langchain 库冲突的 segfault）

用法（命令行）：
    D:/Anaconda3/envs/graph/python.exe scripts/rag_query.py faq "7天无理由退货" 3
    D:/Anaconda3/envs/graph/python.exe scripts/rag_query.py products "推荐游戏本" 3

输出：
    stdout 是 JSON 字符串，格式：[{"score": float, "meta": dict}, ...]
    错误时 stderr 输出，stdout 输出空

为什么需要这个文件：
    在 langchain 进程里直接 import faiss + sentence_transformers 会 segfault
    （参见 memory/openpyxl-sentence-transformers-segfault.md）
    拆成子进程：每个 query 一个干净的 Python 进程，RAG 库单独加载，避开冲突
"""
import importlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "用法: rag_query.py <faq|products> <query> [top_k]"}, ensure_ascii=False))
        sys.exit(1)

    name = sys.argv[1]
    query = sys.argv[2]
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    if name not in ("faq", "products"):
        print(json.dumps({"error": f"未知索引: {name}"}, ensure_ascii=False))
        sys.exit(1)

    # 把 retriever 内部的 print 全部转到 stderr，保住 stdout 只输出 JSON
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        from src.rag.retriever import FaissRetriever
        retriever = FaissRetriever(name)
        hits = retriever.retrieve(query, top_k=top_k)
    finally:
        sys.stdout = old_stdout

    # 只把 JSON 写到 stdout
    print(json.dumps(hits, ensure_ascii=False))


if __name__ == "__main__":
    main()
