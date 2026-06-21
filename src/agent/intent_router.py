"""
意图路由 - S5 4 重保险架构（对齐 PROJECT_PLAN.md §6.3）
- 第 1 重保险：本地 Qwen2.5-0.5B + LoRA 意图分类
- 第 2 重保险：关键词规则后处理（修 product_intro ⇄ policy_qa 互混）
- 第 3 重保险：置信度低时调云端 qwen-flash 重新分类（可选，llm=None 时跳过）
- 第 4 重保险：客服人工审核（前端 UI 层，本模块不涉及）

工作流：
    用户问 → 第 1 重 → 第 2 重 → (第 3 重 if conf<0.7) → intent → 路由到 tool
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage, SystemMessage

from src.intent.classifier import classify_with_confidence, INTENTS, CONFIDENCE_THRESHOLD
from src.intent.postprocess import postprocess


# ============ 意图 → tool 路由表 ============
INTENT_TO_TOOL = {
    "order_query":        "query_order",
    "product_intro":      "search_product",
    "product_recommend":  "search_product",
    "policy_qa":          "search_faq",
    "ticket_transfer":    None,  # 不调 tool，直接返回"转人工"
}


def classify_and_route(user_query: str, llm=None) -> dict:
    """意图识别 + 路由（4 重保险）

    Args:
        user_query: 用户原始问题
        llm: 可选云端 LLM，传了就启用第 3 重保险（云端复核）

    Returns:
        {
            "intent": str,         # 最终意图
            "confidence": float,   # 置信度
            "tool": str | None,    # 路由到的 tool 名
            "was_changed": bool,   # 是否被第 2 重保险（规则）改判
            "reason": str,         # 调试用：触发原因
        }
    """
    # 第 1 重保险：本地意图模型（Qwen2.5-0.5B + LoRA）
    intent, conf, raw = classify_with_confidence(user_query)

    # 第 2 重保险：关键词规则（修 product_intro ⇄ policy_qa 互混）
    intent, was_changed, reason = postprocess(user_query, intent, conf)

    # 第 3 重保险：云端复核（仅在 conf 低 + 没被第 2 重改判 + llm 给了 才调）
    if (
        llm is not None
        and conf < CONFIDENCE_THRESHOLD
        and not was_changed
    ):
        try:
            fallback_intent = _qwen3_max_reclassify(user_query, llm)
            if fallback_intent:
                intent = fallback_intent
        except Exception as e:
            # 第 3 重失败不影响主流程，但需要留痕（云端 key 失效/超时/限流时方便排查）
            print(f"[intent_router] L3 fallback failed: {type(e).__name__}: {e}", file=sys.stderr)

    return {
        "intent": intent,
        "confidence": conf,
        "tool": INTENT_TO_TOOL.get(intent),
        "was_changed": was_changed,
        "reason": reason,
    }


def _qwen3_max_reclassify(user_query: str, llm) -> str | None:
    """第 3 重保险：调云端 LLM 重新分类。失败返回 None。"""
    prompt = (
        f"对用户问题进行意图分类，输出以下之一：{', '.join(INTENTS)}\n"
        f"只输出 intent 名称，不要解释。"
    )
    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=user_query),
    ])
    text = (response.content or "").strip().lower()
    for intent in INTENTS:
        if intent in text:
            return intent
    return None


if __name__ == "__main__":
    """冒烟测试：5 个 query 验证路由（不传 llm，跳过第 3 重保险）"""
    print("=== 意图路由冒烟测试 ===\n")

    test_cases = [
        "我的订单 JD20240610001 到哪了？",
        "激活后能退吗？",
        "推荐一款游戏本",
        "我要投诉转人工",
        "三包政策是什么？",
    ]

    for q in test_cases:
        result = classify_and_route(q)
        print(f"Q: {q}")
        print(f"  → intent={result['intent']}, conf={result['confidence']:.3f}")
        print(f"  → tool={result['tool']}, changed={result['was_changed']}")
        print(f"  → reason={result['reason']}")
        print()
