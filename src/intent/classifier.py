"""
意图分类器 - S5 Agent 入口
- 加载 Qwen2.5-0.5B + LoRA (template=qwen, 与训练严格对齐)
- 提供 classify() / classify_with_confidence() 两种调用
- 低置信度时由调用方触发第 3 重保险（调云端 LLM 重新分类）

⚠️ 与 test/test_intent.py 的 classify() 行为一致，
   后续 S5 推进时可统一改用本模块，test_intent.py 退化为评测脚本。
"""
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASE_MODEL_PATH = PROJECT_ROOT / "models" / "Qwen2.5-0.5B"
LORA_PATH = PROJECT_ROOT / "saves" / "qwen2.5-0.5b-intent-lora"

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

# 训练时用的 instruction，与推理端必须一致
SYSTEM_PROMPT = "对用户问题进行意图分类，输出以下之一：order_query, product_intro, product_recommend, policy_qa, ticket_transfer"

# 置信度阈值：低于此值建议触发第 3 重保险（云端 LLM 重新分类）
# S7 评测时可微调；建议起步 0.7
CONFIDENCE_THRESHOLD = 0.7

_model = None
_tokenizer = None


def _load_model():
    """单例模式加载模型 + LoRA"""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    _tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH, trust_remote_code=True
    )
    _model = AutoModelForCausalLM.from_pretrained(
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
    _model = PeftModel.from_pretrained(_model, lora_adapter)
    _model.eval()
    return _model, _tokenizer


def classify(query: str) -> str:
    """返回意图标签（与 test_intent.py classify() 行为一致）"""
    label, _, _ = classify_with_confidence(query)
    return label


def classify_with_confidence(query: str):
    """返回 (intent_label, confidence, raw_token)

    confidence = softmax(logits) 在第一个生成 token 上的最大概率
    未匹配到 5 类时 intent 为 None，confidence 仍返回（用于人工判断）

    第 3 重保险建议：
        if confidence < CONFIDENCE_THRESHOLD or intent is None:
            call qwen_flash_reclassify(query)  # 见 intent_router.py
    """
    model, tokenizer = _load_model()

    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        {"role": "user", "content": SYSTEM_PROMPT + "\n" + query},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # 第一个生成位置的 vocab 维 logits
        next_token_logits = outputs.logits[0, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)
        top_prob, top_token_id = probs.max(dim=-1)
        top_token = tokenizer.decode([top_token_id])

    # 归一化匹配
    predicted_normalized = top_token.strip().lower().replace("_", "")
    intent = None
    for i in INTENTS:
        if i.replace("_", "") in predicted_normalized:
            intent = i
            break

    return intent, float(top_prob.item()), top_token


if __name__ == "__main__":
    # 简单冒烟
    samples = [
        "我的订单JD20240510001签收了吗？",
        "这款笔记本多大内存？",
        "推荐一款 5000 以内的游戏本",
        "耳机激活后能退吗？",
        "我要投诉，转人工！",
    ]
    print("冒烟测试 (label, confidence):")
    for q in samples:
        intent, conf, raw = classify_with_confidence(q)
        print(f"  {q:<30} → {intent or 'None':<18} conf={conf:.3f}  raw={raw!r}")
