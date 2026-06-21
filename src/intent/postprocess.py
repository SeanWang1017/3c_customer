"""
规则后处理 - 第 2 重保险
- 基于关键词的意图修正
- 解决 product_intro ⇄ policy_qa 互混等已知弱项
- 不调 LLM，纯本地规则，零延迟

工作流：
  1. classifier.classify_with_confidence() 拿到 (intent, confidence)
  2. 若 confidence < threshold 或意图在易混集中，调用 postprocess()
  3. postprocess() 用正则匹配关键词，必要时改判

可调参数：threshold（默认 0.7，与 classifier 一致）
"""
import re
from typing import Optional

# 关键词规则：低置信度或易混意图时生效
# 格式: (pattern, override_intent, reason)
# 顺序敏感：先匹配先生效
RULES = [
    # === product_intro 关键词：问这个具体产品的属性 ===
    (
        r"这[款台个](.{0,8})?(保修|续航|参数|配置|规格|内存|存储|容量|屏幕|刷新率|分辨率|电池|重量|颜色|型号|摄像头|像素|CPU|GPU|显卡|处理器|芯片|接口|材质|尺寸|支持|防水|快充)",
        "product_intro",
        "具体产品属性问句",
    ),
    (
        r"这[款台个].{0,5}(保修|激活|还能|可以)",
        "product_intro",
        "产品保修/激活（带产品上下文）",
    ),
    (
        r"^(多大|几寸|几G|多少G|几核|什么颜色|有几种|支持.{0,3}吗)",
        "product_intro",
        "产品参数速问",
    ),

    # === policy_qa 关键词：通用政策 ===
    (
        r"(7天|七天|15天|无理由|三包|发票|国行|港版|以旧换新|学生|教育优惠|保修政策|退换货政策).{0,4}(退|换|修|开发票|享受|算不算|怎么|多久|能|可以|适用)",
        "policy_qa",
        "通用售后政策",
    ),
    (
        r"(激活后|拆封后|使用后|过了.{0,3}天).{0,3}(退|换|修|保修)",
        "policy_qa",
        "激活后政策",
    ),
    (
        r"(^保修|保修期|保修多久|保修几年|怎么保修)",
        "policy_qa",
        "通用保修政策（无产品上下文）",
    ),
    (
        r"^三包.{0,8}(是什么|什么意思|包括|内容|怎么|适用|哪些|和.{0,4}区别)",
        "policy_qa",
        "三包政策定义/范围",
    ),
    (
        r"^(什么是|什么叫)(三包|保修|发票|售后|七天|无理由)",
        "policy_qa",
        "政策定义类问句",
    ),

    # === order_query 关键词：订单/物流 ===
    (
        r"(JD\d+|订单号?\s*\d|快递\s*\d|单号\s*\d|运单\s*\d)",
        "order_query",
        "订单/快递号",
    ),
    (
        r"(我的)?(订单|快递|包裹|物流|发货|派送|签收).{0,6}(查|状态|多久|什么时候|到哪|信息|进度|到没)",
        "order_query",
        "订单/物流状态",
    ),

    # === product_recommend 关键词：求推荐/有预算 ===
    (
        r"(推荐|求推荐|选哪款|哪款好|选什么|买什么|挑哪|哪.{0,2}适合)",
        "product_recommend",
        "推荐/选购意图",
    ),
    (
        r"\d{3,5}\s*(元|块|以内|以下|上下|左右)",
        "product_recommend",
        "带预算（可能求推荐）",
    ),
    (
        r"(送|给).{0,6}(女朋友|男朋友|爸妈|父母|老人|孩子|小孩|同事|领导)",
        "product_recommend",
        "送礼推荐",
    ),

    # === ticket_transfer 关键词：情绪/转人工 ===
    (
        r"(投诉|转人工|找人工|经理|主管|差评|退款不|太烂|态度差|气死|垃圾|解决不了|没人|升级|曝光)",
        "ticket_transfer",
        "情绪/转人工信号",
    ),
]


def should_apply_rules(intent: Optional[str], confidence: float, threshold: float = 0.7) -> bool:
    """是否需要应用规则后处理

    触发条件：
    - 置信度低于阈值（模型不确定）
    - OR 意图在已知易混集中（product_intro / policy_qa 互混高发）
    - OR 意图为 None（模型输出未匹配）
    """
    if intent is None:
        return True
    if confidence < threshold:
        return True
    if intent in {"product_intro", "policy_qa"}:
        return True
    return False


def postprocess(
    query: str,
    intent: Optional[str],
    confidence: float,
    threshold: float = 0.7,
):
    """应用规则后处理

    Args:
        query: 原始用户问题
        intent: classifier 给出的意图
        confidence: classifier 给出的置信度
        threshold: 置信度阈值

    Returns:
        (corrected_intent, was_changed, reason)
        - corrected_intent: 最终意图（可能与 intent 相同）
        - was_changed: 是否被规则改判
        - reason: 触发原因（调试用）
    """
    if not should_apply_rules(intent, confidence, threshold):
        return intent, False, "no need (high conf & not confused)"

    for pattern, override_intent, reason in RULES:
        if re.search(pattern, query):
            if override_intent != intent:
                return override_intent, True, f"rule override: {reason}"
            return intent, False, f"rule confirmed: {reason}"

    return intent, False, "no rule matched"


if __name__ == "__main__":
    # 冒烟测试：用 known 弱项样本验证
    test_cases = [
        # (query, model_intent, model_conf) → expected_override
        ("这款笔记本保修多久", "policy_qa", 0.55),  # 应改 product_intro
        ("保修期多久", "product_intro", 0.50),  # 应改 policy_qa
        ("7天无理由怎么算", "product_intro", 0.60),  # 应改 policy_qa
        ("JD20240510001 签收了吗", "policy_qa", 0.45),  # 应改 order_query
        ("3000元以内买什么手机", "policy_qa", 0.65),  # 应改 product_recommend
        ("我要投诉转人工", "order_query", 0.40),  # 应改 ticket_transfer
        ("显卡什么型号", "product_intro", 0.92),  # 高置信，不动
    ]
    print("规则后处理冒烟测试:")
    print(f"  {'Query':<25} {'Model':<18} {'Conf':>5} → {'Final':<18} Changed? Reason")
    for q, mi, mc in test_cases:
        final, changed, reason = postprocess(q, mi, mc)
        marker = "✅" if changed else "  "
        print(f"  {q:<25} {mi:<18} {mc:>5.2f} → {final or 'None':<18} {marker}  {reason}")
