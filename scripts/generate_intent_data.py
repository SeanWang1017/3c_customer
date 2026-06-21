"""
从 data/train.jsonl 生成意图分类训练数据
- 输入：每条数据的第一条 user 消息
- 输出：scenario 标签

格式：alpaca（兼容 LLaMA Factory）
"""

import json
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INTENTS = [
    "order_query",
    "product_intro",
    "product_recommend",
    "policy_qa",
    "ticket_transfer",
]

INSTRUCTION = "对用户问题进行意图分类，输出以下之一：order_query, product_intro, product_recommend, policy_qa, ticket_transfer"


def convert(input_path: Path, output_path: Path):
    samples = []
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            messages = data.get("messages", [])
            scenario = data.get("scenario", "").strip()

            if not messages or not scenario:
                skipped += 1
                continue

            if scenario not in INTENTS:
                skipped += 1
                continue

            # 取第一条 user 消息
            first_user = None
            for m in messages:
                if m.get("role") == "user":
                    first_user = m.get("content", "").strip()
                    break

            if not first_user or len(first_user) < 3:
                skipped += 1
                continue

            samples.append({
                "instruction": INSTRUCTION,
                "input": first_user,
                "output": scenario,
            })

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 统计
    label_counts = Counter(s["output"] for s in samples)
    print(f"输入: {input_path.name}")
    print(f"输出: {output_path.name}")
    print(f"总样本: {len(samples)} 条（跳过 {skipped} 条）")
    print(f"标签分布:")
    for label in INTENTS:
        n = label_counts.get(label, 0)
        pct = n / len(samples) * 100 if samples else 0
        print(f"  {label}: {n} 条 ({pct:.1f}%)")


if __name__ == "__main__":
    data_dir = PROJECT_ROOT / "data"

    convert(data_dir / "train.jsonl", data_dir / "intent_classify_train.jsonl")
    print()
    convert(data_dir / "val.jsonl", data_dir / "intent_classify_val.jsonl")
    print()
    convert(data_dir / "test.jsonl", data_dir / "intent_classify_test.jsonl")
