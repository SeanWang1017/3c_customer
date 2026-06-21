"""B 线最小端到端测试：会话 C 转人工 → create_ticket → n8n → 邮件"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.customer_service_agent import (
    build_llm, build_agent, format_history_for_ticket, extract_order_id_from_history
)
from src.agent.intent_router import classify_and_route
from src.tools.ticket import create_ticket
from langchain_core.messages import HumanMessage

print("=== B 线端到端：转人工 → n8n → 163 邮件 ===\n")

llm = build_llm()
agent = build_agent(llm=llm)
thread_id = "user_complaint_smoke"

# 第 1 轮：先建立上下文（订单号 + 商品名进对话历史）
print("第 1 轮 → 用户报告问题")
config = {"configurable": {"thread_id": thread_id}}
agent.invoke(
    {"messages": [HumanMessage(content="我的订单 JD20240610003 雷蛇耳机有问题")]},
    config=config,
)
print("  ✅ 第 1 轮已入会话历史\n")

# 第 2 轮：投诉 query → ticket_transfer 意图
user_query = "我要投诉！找客服三次都没解决，要求经理出面赔偿！"
print(f"第 2 轮 → {user_query}")

route = classify_and_route(user_query, llm=llm)
print(f"  意图: {route['intent']}, conf={route['confidence']:.3f}, was_changed={route['was_changed']}")
assert route["intent"] == "ticket_transfer", f"意图分类错误，得到 {route['intent']}"

# 从 LangGraph state 提取历史
state = agent.get_state(config)
prior = state.values.get("messages", []) if state and state.values else []
print(f"  state 中已有消息 {len(prior)} 条")

all_msgs = prior + [HumanMessage(content=user_query)]
history_text = format_history_for_ticket(all_msgs)
order_id = extract_order_id_from_history(all_msgs)
print(f"  抓到订单号: {order_id or '(无)'}")
print(f"  对话历史预览（前 200 字）：\n    {history_text[:200]}\n")

# 调 create_ticket
result = create_ticket(
    user_id=thread_id,
    user_query=user_query,
    conversation_history=history_text,
    order_id=order_id,
)

print(f"\n=== 结果 ===")
print(f"  ticket_id: {result.get('ticket_id')}")
print(f"  notified:  {result.get('notified')}")
print(f"  message:   {result.get('message')}")
print(f"\n→ 请检查 QQ 邮箱是否收到新邮件（subject 含 {result.get('ticket_id', '?')}）")
