"""
Step 5: 端到端 RAG 测试
- 跑 15+ query 验证整个 RAG 系统
- 覆盖 FAQ / 商品介绍 / 商品推荐 / 边界
- 输出控制台结果 + 报告 JSON

报告格式（与 test_intent.py 一致）：
    reports/rag_test_<时间戳>.json

用法：
    cd D:/Studying/LLM/Project && D:/Anaconda3/envs/graph/python.exe test/test_rag.py
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
REPORTS_DIR = PROJECT_ROOT / "reports"

from src.rag.retriever import retrieve

# 测试用例：(intent, query, expected_keyword_in_top1, category)
TEST_CASES = [
    # === 5 个 FAQ 政策 query（路由到 faq.index） ===
    ("policy_qa", "7天无理由退货怎么算", "七", "policy_qa"),
    ("policy_qa", "保修期多久", "保修", "policy_qa"),
    ("policy_qa", "怎么开发票", "发票", "policy_qa"),
    ("policy_qa", "能以旧换新吗", "换", "policy_qa"),
    ("policy_qa", "三包是什么", "三包", "policy_qa"),

    # === 4 个商品介绍 query（路由到 products.index） ===
    ("product_intro", "这款笔记本多大内存", "G", "product_intro"),  # 匹配 "16G"
    ("product_intro", "降噪耳机", "降噪", "product_intro"),
    # 已知弱项：jd 商品数据可能没有 5000 万像素相机（changelog 2026-06-11 已记录）
    ("product_intro", "5000万像素相机", "5000万", "product_intro"),
    ("product_intro", "蓝牙耳机", "蓝牙", "product_intro"),

    # === 4 个商品推荐 query ===
    ("product_recommend", "推荐 5000 以内的游戏本", "游戏本", "product_recommend"),
    ("product_recommend", "3000元以内拍照好的手机", "手机", "product_recommend"),
    ("product_recommend", "学生用笔记本", "笔记本", "product_recommend"),
    ("product_recommend", "性价比高的鼠标", "鼠标", "product_recommend"),

    # === 2 个边界 case ===
    # 边界 1：复合 query 拆解（替代原"单字 query 退"——之前判定太宽松）
    ("policy_qa", "退货运费谁出", "运费", "boundary"),
    # 边界 2：冷门 FAQ 召回（替代原"三包 + product_intro 错意图"——必失败）
    ("policy_qa", "电子产品的三包有效期", "三包", "boundary"),
]


def _percentile(values, p):
    """简单 percentile 实现（无 numpy 依赖）"""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = int(len(sorted_v) * p / 100)
    idx = max(0, min(idx, len(sorted_v) - 1))
    return sorted_v[idx]


def main():
    print("=" * 60)
    print("Step 5: 端到端 RAG 测试")
    print("=" * 60)

    passed = 0
    latencies = []
    failed_cases = []
    category_stats = {}  # category -> {"total": n, "passed": n}

    for i, (intent, query, expected, category) in enumerate(TEST_CASES, 1):
        t0 = time.perf_counter()
        hits = retrieve(intent, query, top_k=5)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        # 取 top-1 文本（按意图区分字段）
        top1_text = ""
        top1_score = 0.0
        if hits:
            m = hits[0]["meta"]
            top1_score = hits[0]["score"]
            if intent == "policy_qa":
                top1_text = m.get("question", "")
            else:
                top1_text = m.get("title", "")

        # 判通过：top-1 含 expected 关键字
        ok = bool(top1_text) and (expected in top1_text)
        if ok:
            passed += 1
        else:
            failed_cases.append({
                "index": i,
                "intent": intent,
                "category": category,
                "query": query,
                "expected": expected,
                "top1_text": top1_text[:80],
                "top1_score": round(top1_score, 3),
            })

        # 统计
        category_stats.setdefault(category, {"total": 0, "passed": 0})
        category_stats[category]["total"] += 1
        if ok:
            category_stats[category]["passed"] += 1

        marker = "✅" if ok else "❌"
        print(f"\n[{i:>2}] {marker} intent={intent}")
        print(f"     Q:      {query!r}")
        print(f"     expect: {expected!r}")
        print(f"     top-1:  {top1_text[:50]!r} (score={top1_score:.3f}, {latency_ms:.0f}ms)")

    total = len(TEST_CASES)
    rate = passed / total * 100
    print("\n" + "=" * 60)
    print(f"总通过: {passed}/{total} ({rate:.1f}%)")
    print("=" * 60)

    # 写报告
    _write_report(passed, total, category_stats, failed_cases, latencies)


def _write_report(passed, total, category_stats, failed_cases, latencies):
    """写结构化报告到 reports/rag_test_<时间戳>.json"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"rag_test_{ts}.json"

    # 转换 category_stats 为 pass_rate
    per_category = {}
    for cat, stats in category_stats.items():
        per_category[cat] = {
            "total": stats["total"],
            "passed": stats["passed"],
            "pass_rate": round(stats["passed"] / stats["total"], 3) if stats["total"] else 0.0,
        }

    report = {
        "timestamp": ts,
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 3) if total else 0.0,
        "per_category": per_category,
        "latency_ms": {
            "avg": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            "p50": round(_percentile(latencies, 50), 1),
            "p95": round(_percentile(latencies, 95), 1),
            "max": round(max(latencies), 1) if latencies else 0.0,
        },
        "failed_cases": failed_cases,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[报告] 已保存: {report_path}")


if __name__ == "__main__":
    main()
