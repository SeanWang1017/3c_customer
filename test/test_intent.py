"""
意图分类测试脚本
- 加载 Qwen2.5-0.5B + LoRA 微调模型
- 支持两种模式：
  - smoke: 40 条手写用例（快速冒烟测试，每类 8 条）
  - full:  245 条真实测试集（统计性评估，含 P/R/F1 + 混淆矩阵）
- full 模式报告输出到 reports/intent_eval_<时间戳>.json

训练-推理格式链路（必须严格对齐，否则准确率会塌掉）：
  当前:  1. 数据文件: data/intent_classify_*.jsonl  (alpaca 格式: instruction/input/output)
        2. LLaMA Factory template=qwen (见 config/train_lora.yaml)
        3. 训练 token 化: system="You are Qwen..." + user=instruction+input
        4. 推理 (本脚本) : 用相同 qwen chat template，见 classify()
  TODO:  下次训练计划改用 template=default (LLaMA Factory 默认模板)，
        届时必须同步修改 classify() 的 messages 格式 / 是否保留 chat template
        —— 等新模型训完一起改。

用法：
    conda activate graph
    python test/test_intent.py                # 默认 full 模式
    python test/test_intent.py --mode smoke   # 快速冒烟
    python test/test_intent.py --no-report    # full 模式不写文件
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASE_MODEL_PATH = PROJECT_ROOT / "models" / "Qwen2.5-0.5B"
LORA_PATH = PROJECT_ROOT / "saves" / "qwen2.5-0.5b-intent-lora"
TEST_SET_PATH = PROJECT_ROOT / "data" / "intent_classify_test.jsonl"
REPORTS_DIR = PROJECT_ROOT / "reports"

INTENTS = [
    "order_query",
    "product_intro",
    "product_recommend",
    "policy_qa",
    "ticket_transfer",
]

INTENT_LABELS = {
    "order_query": "订单查询",
    "product_intro": "商品介绍",
    "product_recommend": "商品推荐",
    "policy_qa": "政策问答",
    "ticket_transfer": "转人工",
}

# 训练时使用的 instruction，推理时必须保持一致
SYSTEM_PROMPT = "对用户问题进行意图分类，输出以下之一：order_query, product_intro, product_recommend, policy_qa, ticket_transfer"

TEST_QUERIES = {
    "order_query": [
        "我的订单JD20240510001签收了吗？",
        "快递今天能到吗？",
        "帮我查一下订单 123456",
        "查一下我的快递到哪了",
        "JD202605075389 这个订单什么时候发货",
        "物流信息能发我一下吗",
        "我的包裹显示派送中还要多久",
        "能帮我看看订单 10086 的状态吗",
    ],
    "product_intro": [
        "这款笔记本多大内存？",
        "存储空间多大？",
        "显卡什么型号？",
        "这款手机支持 5G 吗",
        "有几种颜色",
        "屏幕刷新率多少",
        "重量多少",
        "摄像头像素多少",
    ],
    "product_recommend": [
        "推荐一款 5000 以内的游戏本",
        "学生党电脑求推荐",
        "便宜好用的耳机有吗？",
        "送女朋友什么手机好",
        "3000 以内拍照好的手机",
        "商务笔记本推荐",
        "适合画画的平板",
        "性价比高的路由器",
    ],
    "policy_qa": [
        "耳机激活后能退吗？",
        "三包政策是什么？",
        "能开发票吗？",
        "7 天无理由是收到第几天算",
        "保修期多久",
        "发票丢了能补吗",
        "以旧换新怎么操作",
        "学生有教育优惠吗",
    ],
    "ticket_transfer": [
        "我要投诉，转人工！",
        "问题拖了一周没人处理",
        "太气人了，我要升级投诉！",
        "你们客服态度太差了",
        "我要见你们经理",
        "找真人",
        "这个问题解决不了我只能投诉了",
        "人工客服在哪",
    ],
}


def load_model():
    """加载基础模型 + LoRA"""
    print("=" * 50)
    print("加载模型...")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH, trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    checkpoints = sorted(
        [p for p in LORA_PATH.iterdir() if p.name.startswith("checkpoint-")],
        key=lambda x: int(x.name.split("-")[-1]),
    )
    lora_adapter = checkpoints[-1] if checkpoints else LORA_PATH
    print(f"  使用 LoRA: {lora_adapter.name}")
    model = PeftModel.from_pretrained(model, lora_adapter)
    model.eval()
    return model, tokenizer


def classify(model, tokenizer, query: str) -> str:
    """使用 qwen chat template 进行意图分类

    必须与 LLaMA Factory 训练时的 template=qwen 格式完全一致：
        训练 token 化 = system("You are Qwen...") + user(instruction+input)
        推理 (本函数) = 同样的 system + user 结构，再 add_generation_prompt

    训练数据本身是 alpaca 格式 (jsonl with instruction/input/output)，
    是 LLaMA Factory 在 template=qwen 下转成上面那个 chat 结构喂给模型的。
    """
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": SYSTEM_PROMPT + "\n" + query},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    raw = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    # 取第一个词作为标签
    first_word = raw.strip().split()[0] if raw.strip() else ""
    return first_word


def parse_intent(predicted: str):
    """归一化匹配模型输出到 5 类之一；未匹配返回 None"""
    predicted_normalized = predicted.replace("_", "").lower()
    for intent in INTENTS:
        if intent.replace("_", "").lower() in predicted_normalized:
            return intent
    return None


def load_test_set(path: Path):
    """从 jsonl 读取 [(query, label), ...]"""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            samples.append((item["input"], item["output"]))
    return samples


def compute_metrics(preds, labels):
    """计算总准确率、每类 P/R/F1、5x5 混淆矩阵 + 未匹配计数"""
    total = len(preds)
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    accuracy = correct / total if total else 0.0

    # 5 类内混淆；模型未匹配输出单独计数
    confusion = {t: {p: 0 for p in INTENTS} for t in INTENTS}
    no_match_count = 0
    for t, p in zip(labels, preds):
        if p is None:
            no_match_count += 1
            continue
        if t in confusion and p in confusion[t]:
            confusion[t][p] += 1

    per_intent = {}
    for intent in INTENTS:
        tp = confusion[intent][intent]
        fp = sum(confusion[t][intent] for t in INTENTS if t != intent)
        fn = sum(confusion[intent][p] for p in INTENTS if p != intent)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        support = sum(confusion[intent].values())
        per_intent[intent] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_intent": per_intent,
        "confusion": confusion,
        "no_match_count": no_match_count,
    }


def print_metrics(metrics, latencies=None):
    """打印总准确率 + 每类 P/R/F1 + 混淆矩阵 + 误分类对 Top 列表"""
    accuracy = metrics["accuracy"]
    per_intent = metrics["per_intent"]
    confusion = metrics["confusion"]

    print("\n" + "=" * 64)
    print(
        f"[总] 准确率: {accuracy * 100:.1f}% "
        f"({metrics['correct']}/{metrics['total']})"
    )
    if latencies:
        print(
            f"[耗时] 平均 {latencies['avg'] * 1000:.0f}ms | "
            f"P50 {latencies['p50'] * 1000:.0f}ms | "
            f"P95 {latencies['p95'] * 1000:.0f}ms | "
            f"Max {latencies['max'] * 1000:.0f}ms"
        )
    if metrics["no_match_count"] > 0:
        print(f"[警告] 未匹配样本: {metrics['no_match_count']} (模型输出不在 5 类内)")
    print("=" * 64)

    # 每类指标
    print("\n每类详细指标:")
    print(f"  {'意图':<20} {'精确率':>8} {'召回率':>8} {'F1':>8} {'样本':>6}")
    print("  " + "-" * 54)
    for intent in INTENTS:
        m = per_intent[intent]
        cn = INTENT_LABELS.get(intent, intent)
        print(
            f"  {cn} ({intent})".ljust(22)
            + f"  {m['precision']:>7.3f}  {m['recall']:>7.3f}  "
            f"{m['f1']:>7.3f}  {m['support']:>5d}"
        )

    # 混淆矩阵
    short = {
        "order_query": "order_q",
        "product_intro": "prod_i",
        "product_recommend": "prod_r",
        "policy_qa": "policy",
        "ticket_transfer": "ticket",
    }
    print("\n混淆矩阵 (行=真实, 列=预测):")
    header = "  " + " " * 20 + "".join(f"{short[i]:>8}" for i in INTENTS)
    print(header)
    for t in INTENTS:
        cn = INTENT_LABELS.get(t, t)
        row = f"  {cn} ({t})".ljust(22)
        for p in INTENTS:
            row += f"{confusion[t][p]:>8d}"
        print(row)

    # 误分类对 Top
    mis_pairs = []
    for t in INTENTS:
        for p in INTENTS:
            if t != p and confusion[t][p] > 0:
                mis_pairs.append((confusion[t][p], t, p))
    if mis_pairs:
        print("\n最常见的误分类对 (Top 5):")
        for n, t, p in sorted(mis_pairs, reverse=True)[:5]:
            print(
                f"  {n:>3} 次: {INTENT_LABELS[t]} → 被预测为 {INTENT_LABELS[p]}"
            )


def run_smoke(model, tokenizer):
    """40 条手写用例的快速冒烟测试（保留原行为）"""
    print("\n" + "=" * 50)
    print("【冒烟测试】40 条手写用例")
    print("=" * 50)
    total = 0
    correct = 0
    failed_cases = []

    for expected_intent, queries in TEST_QUERIES.items():
        print(f"\n【{INTENT_LABELS[expected_intent]} ({expected_intent})】")
        print("-" * 50)

        for query in queries:
            total += 1
            predicted_raw = classify(model, tokenizer, query)
            predicted_intent = parse_intent(predicted_raw)

            is_correct = predicted_intent == expected_intent
            if is_correct:
                correct += 1
            else:
                failed_cases.append((query, expected_intent, predicted_raw))

            status = "✅" if is_correct else "❌"
            print(f"  {status} Q: {query}")
            print(
                f"     预期: {expected_intent} | "
                f"实际: {predicted_intent or '(未匹配)'} | "
                f"原始: {predicted_raw!r}"
            )

    print("\n" + "=" * 50)
    print(f"冒烟结果: {correct}/{total} 正确 ({correct / total * 100:.1f}%)")
    print("=" * 50)

    if failed_cases:
        print("\n失败用例:")
        for q, expected, predicted in failed_cases:
            print(f"  ❌ Q: {q}")
            print(f"     预期: {expected} | 实际: {predicted}")


def run_full(model, tokenizer, save_report: bool):
    """加载 245 条真实测试集，输出 P/R/F1 + 混淆矩阵 + 耗时"""
    print("\n" + "=" * 50)
    print(f"【全量评估】加载 {TEST_SET_PATH.name}")
    print("=" * 50)

    samples = load_test_set(TEST_SET_PATH)
    print(f"  共加载 {len(samples)} 条样本")

    preds = []
    labels = []
    latencies = []
    failed_cases = []

    for i, (query, label) in enumerate(samples, 1):
        t0 = time.perf_counter()
        predicted_raw = classify(model, tokenizer, query)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        predicted_intent = parse_intent(predicted_raw)
        preds.append(predicted_intent)
        labels.append(label)

        if predicted_intent != label:
            failed_cases.append(
                (query, label, predicted_intent, predicted_raw)
            )

        if i % 25 == 0 or i == len(samples):
            print(f"\r  推理进度: {i}/{len(samples)}", end="", flush=True)
    print()

    # 延迟统计
    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    latency_stats = {
        "avg": sum(latencies) / n,
        "p50": sorted_lat[n // 2],
        "p95": sorted_lat[int(n * 0.95)],
        "max": sorted_lat[-1],
    }

    metrics = compute_metrics(preds, labels)
    print_metrics(metrics, latency_stats)

    if failed_cases:
        print(f"\n失败用例预览 (共 {len(failed_cases)} 条, 仅显示前 5):")
        for q, expected, predicted, raw in failed_cases[:5]:
            print(f"  ❌ Q: {q}")
            print(
                f"     预期: {expected} | "
                f"实际: {predicted or '(未匹配)'} | "
                f"原始: {raw!r}"
            )

    if save_report:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = REPORTS_DIR / f"intent_eval_{ts}.json"
        report = {
            "timestamp": ts,
            "model": str(LORA_PATH),
            "test_set": str(TEST_SET_PATH),
            "total": metrics["total"],
            "correct": metrics["correct"],
            "accuracy": round(metrics["accuracy"], 4),
            "per_intent": {
                intent: {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in m.items()
                }
                for intent, m in metrics["per_intent"].items()
            },
            "confusion": metrics["confusion"],
            "no_match_count": metrics["no_match_count"],
            "latency_ms": {
                k: round(v * 1000, 1) for k, v in latency_stats.items()
            },
            "failed_count": len(failed_cases),
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[报告] 已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="意图分类测试")
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="full",
        help="smoke=40 条手写用例快速冒烟; full=245 条真实测试集 (默认 full)",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="full 模式不写文件到 reports/",
    )
    args = parser.parse_args()

    model, tokenizer = load_model()
    print("模型加载完成\n")

    if args.mode == "smoke":
        run_smoke(model, tokenizer)
    else:
        run_full(model, tokenizer, save_report=not args.no_report)


if __name__ == "__main__":
    main()
