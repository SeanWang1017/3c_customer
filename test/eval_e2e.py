"""
S7 Step 3 - 端到端 5 query 评测脚本

跑 5 条覆盖 5 种意图的 query，端到端验证：
  ① 用户输入 → ② 意图识别 → ③ 工具调用 → ④ LLM 最终回复

自动评分（每条满分 100）：
  - 意图正确    (20 分)：intent == expected_intent
  - 工具调用正确(25 分)：实际调的 tool == expected_tool（无 tool 场景按 expected="none" 计）
  - 关键词命中  (30 分)：3 组关键词，每组命中其中之一得 10 分
  - 人工质量分  (25 分)：跑完后由用户填，先留 None
  - 总分 = 上述四项之和（人工分缺失时显示 "auto: X/75"）

输出：reports/e2e_eval_<时间戳>.json

用法：
    conda activate graph
    python test/eval_e2e.py
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from src.agent.customer_service_agent import build_llm, build_agent
from src.agent.intent_router import classify_and_route


# ============ 5 条端到端测试用例 ============
# 每条用例覆盖一种意图，关键词分 3 组（每组命中其一即得 10 分）
TEST_CASES = [
    {
        "id": 1,
        "query": "帮我查一下订单 ORD20240010 的状态",
        "expected_intent": "order_query",
        "expected_tool": "query_order",
        "keyword_groups": [
            ["ORD20240010", "20240010"],          # 订单号回显
            ["已发货", "未发货", "运输中", "已签收", "已支付", "待付款", "已取消"],  # 状态
            ["元", "¥", "金额", "物流", "快递"],   # 订单详情
        ],
    },
    {
        "id": 2,
        "query": "耳机已经拆封了还能 7 天无理由退货吗？",
        "expected_intent": "policy_qa",
        "expected_tool": "search_faq",
        "keyword_groups": [
            ["7天", "7 天", "七天", "无理由"],    # 政策
            ["拆封", "包装", "二次销售", "影响"],  # 拆封条件
            ["退货", "退款", "退回"],              # 主题
        ],
    },
    {
        "id": 3,
        "query": "推荐一款 3000 元以内的游戏手机",
        "expected_intent": "product_recommend",
        "expected_tool": "search_product",
        "keyword_groups": [
            ["推荐", "建议", "可以考虑", "如下"],  # 推荐句式
            ["¥", "元", "价格"],                  # 价格信息
            ["游戏", "手机", "性能", "处理器"],    # 商品相关
        ],
    },
    {
        "id": 4,
        "query": "这款雷蛇游戏耳机的降噪效果怎么样？",
        "expected_intent": "product_intro",
        "expected_tool": "search_product",
        "keyword_groups": [
            ["雷蛇", "Razer"],                    # 品牌
            ["降噪", "ANC", "主动降噪", "隔音"],   # 核心功能
            ["耳机", "音质", "麦克风"],            # 商品类别
        ],
    },
    {
        "id": 5,
        "query": "你们态度太差了！我要投诉，转人工！",
        "expected_intent": "ticket_transfer",
        "expected_tool": "none",  # 转人工场景不调 tool，直接回复
        "keyword_groups": [
            ["人工", "客服"],                     # 转接对象
            ["稍等", "稍候", "为您", "已为", "马上", "立即"],  # 服务用语
            ["投诉", "处理", "工单", "记录", "反馈"],  # 处理动作
        ],
    },
]


def extract_tool_calls(messages):
    """从 agent.invoke 返回的 messages 中提取本轮 tool 调用 + 结果。"""
    tool_calls_info = []
    # 最近一次 AIMessage 的 tool_calls
    recent_tcs = []
    for m in messages[::-1]:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            recent_tcs = m.tool_calls
            break
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    relevant = tool_msgs[-len(recent_tcs):] if recent_tcs else []

    for i, tc in enumerate(recent_tcs):
        tool_calls_info.append({
            "tool_name": tc["name"],
            "tool_args": tc["args"],
            "tool_result_preview": (
                relevant[i].content[:300] if i < len(relevant) else None
            ),
        })
    return tool_calls_info


def score_keywords(reply: str, keyword_groups: list[list[str]]) -> tuple[int, list[bool]]:
    """每组关键词命中其中之一即得 10 分，共 30 分。返回 (得分, 各组是否命中列表)。"""
    hits = []
    for group in keyword_groups:
        hit = any(kw in reply for kw in group)
        hits.append(hit)
    score = sum(hits) * 10
    return score, hits


def run_single_case(case: dict, agent, llm) -> dict:
    """跑一条端到端用例，返回结构化结果（含自动评分）。"""
    print("═" * 70)
    print(f"【Case {case['id']}】 {case['query']}")
    print("═" * 70)

    t0 = time.time()

    # ② 意图识别
    route = classify_and_route(case["query"], llm=llm)
    actual_intent = route["intent"]
    intent_score = 20 if actual_intent == case["expected_intent"] else 0
    print(f"\n② 意图识别")
    print(f"   实际 intent={actual_intent}  conf={route['confidence']:.3f}  was_changed={route['was_changed']}")
    print(f"   期望 intent={case['expected_intent']}  → {'✅' if intent_score else '❌'} {intent_score}/20")

    # ③ 智能体决策（转人工特殊处理：不调 agent）
    tool_calls_info = []
    actual_tool = "none"
    final_reply = ""

    if route["intent"] == "ticket_transfer":
        # 转人工：硬编码回复（与 customer_service_agent.chat 行为一致）
        final_reply = "已为您转接人工客服，请稍等。我们会有专员处理您的投诉，给您带来不便非常抱歉。"
        actual_tool = "none"
    else:
        # 调 Agent
        thread_id = f"e2e_eval_case_{case['id']}"
        config = {"configurable": {"thread_id": thread_id}}
        # 给 LLM 路由提示
        if route["tool"]:
            content = f"[路由提示] 用户意图为 {actual_intent}，应调用 {route['tool']} 工具。\n\n用户问题: {case['query']}"
        else:
            content = case["query"]

        result = agent.invoke(
            {"messages": [HumanMessage(content=content)]},
            config=config,
        )
        tool_calls_info = extract_tool_calls(result["messages"])
        actual_tool = tool_calls_info[0]["tool_name"] if tool_calls_info else "none"
        final_reply = result["messages"][-1].content

    tool_score = 25 if actual_tool == case["expected_tool"] else 0
    print(f"\n③ 智能体决策")
    if tool_calls_info:
        for tc in tool_calls_info:
            args_str = ", ".join(f"{k}={v!r}" for k, v in tc["tool_args"].items())
            print(f"   调用工具: {tc['tool_name']}({args_str})")
    else:
        print(f"   未调工具（直接回复）")
    print(f"   实际 tool={actual_tool}  期望 tool={case['expected_tool']}  → {'✅' if tool_score else '❌'} {tool_score}/25")

    # ④ 结果输出 + 关键词评分
    print(f"\n④ 结果输出")
    print(f"   {final_reply[:300]}{'...' if len(final_reply) > 300 else ''}")

    kw_score, kw_hits = score_keywords(final_reply, case["keyword_groups"])
    print(f"\n关键词命中（每组命中之一得 10 分）")
    for i, (group, hit) in enumerate(zip(case["keyword_groups"], kw_hits)):
        status = "✅" if hit else "❌"
        print(f"   组 {i+1} {group}: {status}")
    print(f"   关键词总分: {kw_score}/30")

    elapsed = time.time() - t0

    auto_score = intent_score + tool_score + kw_score
    print(f"\n📊 本条得分: 自动 {auto_score}/75 + 人工 ?/25 = ?/100")
    print(f"   耗时: {elapsed:.2f}s\n")

    return {
        "id": case["id"],
        "query": case["query"],
        "expected_intent": case["expected_intent"],
        "actual_intent": actual_intent,
        "intent_confidence": round(route["confidence"], 3),
        "intent_was_changed": route["was_changed"],
        "intent_change_reason": route["reason"],
        "intent_score": intent_score,
        "expected_tool": case["expected_tool"],
        "actual_tool": actual_tool,
        "tool_score": tool_score,
        "tool_calls": tool_calls_info,
        "keyword_groups": case["keyword_groups"],
        "keyword_hits": kw_hits,
        "keyword_score": kw_score,
        "manual_quality_score": None,  # 跑完后由用户手动填 0-25
        "auto_score": auto_score,
        "auto_total": 75,
        "total_score": None,  # = auto_score + manual_quality_score
        "final_reply": final_reply,
        "latency_s": round(elapsed, 3),
    }


def main():
    print("\n" + "=" * 70)
    print("S7 Step 3 - 端到端评测（5 query × 4 阶段链路）")
    print("=" * 70 + "\n")

    print("[初始化] 构建 LLM + Agent ...")
    llm = build_llm()
    agent = build_agent(llm=llm)
    print("[初始化] 完成\n")

    results = []
    for case in TEST_CASES:
        result = run_single_case(case, agent, llm)
        results.append(result)

    # ============ 汇总 ============
    n = len(results)
    intent_correct = sum(1 for r in results if r["intent_score"] > 0)
    tool_correct = sum(1 for r in results if r["tool_score"] > 0)
    kw_total = sum(r["keyword_score"] for r in results)
    kw_max = n * 30
    auto_total = sum(r["auto_score"] for r in results)
    auto_max = n * 75
    avg_latency = sum(r["latency_s"] for r in results) / n

    print("\n" + "=" * 70)
    print("📊 端到端评测汇总")
    print("=" * 70)
    print(f"意图正确率:   {intent_correct}/{n} ({intent_correct/n*100:.1f}%)")
    print(f"工具调用正确: {tool_correct}/{n} ({tool_correct/n*100:.1f}%)")
    print(f"关键词命中:   {kw_total}/{kw_max} ({kw_total/kw_max*100:.1f}%)")
    print(f"自动评分:     {auto_total}/{auto_max} ({auto_total/auto_max*100:.1f}%)")
    print(f"平均延迟:     {avg_latency:.2f}s")
    print(f"\n⚠️  人工质量分 (0-25/条) 待你填入 JSON 后重算总分\n")

    # ============ 保存 JSON ============
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = PROJECT_ROOT / "reports" / f"e2e_eval_{timestamp}.json"
    report_path.parent.mkdir(exist_ok=True)

    summary = {
        "timestamp": timestamp,
        "total_queries": n,
        "intent_accuracy": f"{intent_correct}/{n}",
        "intent_accuracy_pct": round(intent_correct / n * 100, 1),
        "tool_accuracy": f"{tool_correct}/{n}",
        "tool_accuracy_pct": round(tool_correct / n * 100, 1),
        "keyword_hit": f"{kw_total}/{kw_max}",
        "keyword_hit_pct": round(kw_total / kw_max * 100, 1),
        "auto_score": f"{auto_total}/{auto_max}",
        "auto_score_pct": round(auto_total / auto_max * 100, 1),
        "manual_quality_avg": None,    # 用户填完后重算
        "final_avg_score": None,       # 用户填完后重算
        "avg_latency_s": round(avg_latency, 3),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "details": results},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"💾 报告已保存: {report_path.relative_to(PROJECT_ROOT)}")
    print(f"\n下一步：")
    print(f"  1. 阅读 5 条 final_reply（在 JSON 的 details 里）")
    print(f"  2. 给每条打 0-25 的人工质量分，填到 manual_quality_score 字段")
    print(f"  3. 我会用这份 JSON + 另两份 JSON 生成 docs/evaluation.md\n")


if __name__ == "__main__":
    main()
