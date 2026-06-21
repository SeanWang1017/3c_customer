"""
订单查询 Agent - S5 真实 LLM 版
- 调阿里云百炼（OpenAI 兼容接口）
- 需要 .env 里设 DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL / DASHSCOPE_MODEL
- 全用真实 LLM 调工具，无 mock
"""
import os
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.agents.middleware import ToolCallLimitMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from src.tools.langchain_tools import query_order,search_faq,search_product
from src.tools.ticket import create_ticket
from src.agent.intent_router import classify_and_route, INTENT_TO_TOOL


SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.txt").read_text(encoding="utf-8")


# ============ 工单转人工辅助函数 ============
_ORDER_ID_PATTERN = re.compile(r"JD\d{6,}", re.IGNORECASE)


def format_history_for_ticket(messages) -> str:
    """把 LangGraph state messages 转成多行可读字符串供 n8n 邮件正文用。

    - 只保留 HumanMessage + AIMessage 的 text 内容
    - 跳过 ToolMessage / SystemMessage / 纯 tool_calls 的空 AIMessage
    - 时间戳用当前时间近似（LangGraph 不记录消息真实时间）
    - 每条最长 200 字符避免邮件超长
    """
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = "用户"
        elif isinstance(m, AIMessage):
            if not (m.content or "").strip():
                continue
            role = "机器人"
        else:
            continue
        ts = datetime.now().strftime("%H:%M")
        content = (m.content or "").strip().replace("\n", " ")[:200]
        lines.append(f"[{ts}] {role}：{content}")
    return "\n".join(lines)


def extract_order_id_from_history(messages) -> str:
    """从对话历史里抓订单号（JD + 6 位以上数字），找不到返回空串。"""
    for m in messages:
        text = getattr(m, "content", "") or ""
        match = _ORDER_ID_PATTERN.search(text)
        if match:
            return match.group().upper()
    return ""

def build_llm(
    temperature: float = 0.6,
    max_tokens: int = 1500,
) -> BaseChatModel:
    """构建指向阿里云百炼的 ChatOpenAI（OpenAI 兼容接口）。

    Args:
        temperature: 生成温度，0-1 之间。客服场景推荐 0.5-0.7
                    （低=更确定但死板，高=更多样但可能跑偏）
        max_tokens: 单次生成最大 token 数。客服回复 200-500 够，工具返回 1000+
                   留余量设 1500
    """
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key or api_key == "your_dashscope_api_key_here":
        raise RuntimeError(
            "DASHSCOPE_API_KEY 未设置或为占位符。请在 .env 里填入真实 key。"
        )
    return ChatOpenAI(
        model=os.getenv("DASHSCOPE_MODEL", "qwen-flash"),
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def build_agent(llm: BaseChatModel | None = None):
    """构建客服 Agent。llm 不传则走 build_llm()。

    特性：
    - 单次 invoke 最多调 5 次 tool（ToolCallLimitMiddleware，防死循环）
    - 多轮对话：用 InMemorySaver checkpointer，按 thread_id 记忆上下文
      调用方在 invoke 时传 config={"configurable": {"thread_id": "xxx"}} 即可
    """
    if llm is None:
        llm = build_llm()

    # 限制：单次 invoke 最多调 5 次工具，超量后 LLM 用已有数据继续生成回复
    tool_limiter = ToolCallLimitMiddleware(
        run_limit=5,
        exit_behavior="continue",
    )

    # 短期记忆：进程内字典存历史，按 thread_id 区分会话
    # 进程重启会丢失。生产环境可换 RedisSaver / SqliteSaver
    checkpointer = InMemorySaver()

    return create_agent(
        model=llm,
        tools=[query_order, search_faq, search_product],
        middleware=[tool_limiter],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )


if __name__ == "__main__":
    print("=== 客服 Agent 多轮对话冒烟测试 ===\n")

    llm = build_llm()
    agent = build_agent(llm=llm)

    def chat(user_query: str, thread_id: str):
        """单轮对话辅助函数：意图路由 → 调 tool → 回复，带 thread_id 记忆。

        端到端 4 阶段可视化：
        ① 用户输入  ② 意图识别  ③ 智能体决策（工具调用细节）  ④ 结果输出
        """
        from langchain_core.messages import ToolMessage

        # ① 用户输入
        print("═" * 60)
        print(f"① 用户输入  [{thread_id}]")
        print(f"   {user_query}")

        # ② 意图识别（第 1 重 + 第 2 重 + 可选第 3 重）
        route = classify_and_route(user_query, llm=llm)
        intent, tool_name = route["intent"], route["tool"]
        print(f"\n② 意图识别  第 1 重本地模型 + 第 2 重规则")
        print(f"   intent={intent}  conf={route['confidence']:.3f}")
        print(f"   → 路由到工具: {tool_name or '无（直接回复）'}")
        if route.get("was_changed"):
            print(f"   [第 2 重改判] {route['reason']}")

        # 路由结果分类处理
        if tool_name is None:
            if intent == "ticket_transfer":
                # ── 转人工：跳过 LLM，直接调 create_ticket（落库 + n8n 通知）
                tt_config = {"configurable": {"thread_id": thread_id}}
                try:
                    state = agent.get_state(tt_config)
                    prior_messages = (
                        state.values.get("messages", []) if state and state.values else []
                    )
                except Exception as e:
                    # 历史拿不到不阻塞工单创建，但留痕（checkpointer 异常 / thread_id 不存在时方便排查）
                    print(f"[agent] get_state failed: {type(e).__name__}: {e}", file=sys.stderr)
                    prior_messages = []

                all_msgs_for_ticket = prior_messages + [HumanMessage(content=user_query)]
                history_text = format_history_for_ticket(all_msgs_for_ticket)
                order_id = extract_order_id_from_history(all_msgs_for_ticket)

                ticket_result = create_ticket(
                    user_id=thread_id,
                    user_query=user_query,
                    conversation_history=history_text,
                    order_id=order_id,
                )

                print(f"\n③ 智能体决策  转人工（直调 create_ticket，跳过 LLM）")
                print(f"   工单ID: {ticket_result.get('ticket_id', '?')}")
                print(
                    f"   n8n 通知: {'✅ 已送达' if ticket_result.get('notified') else '⚠️ 未送达（仅本地落库）'}"
                )
                if order_id:
                    print(f"   关联订单: {order_id}")

                print(f"\n④ 结果输出  [{thread_id}]")
                ticket_id = ticket_result.get("ticket_id", "?")
                print(
                    f"   客服: 已为您创建工单 {ticket_id}，"
                    f"并通知人工客服跟进，请稍候。"
                )
                print("═" * 60 + "\n")
                return
            elif intent is None:
                hint = ""  # 路由失败 → LLM 看历史自主决定
            else:
                print(f"\n③ 智能体决策  无对应工具，直接回复")
                print(f"\n④ 结果输出  [{thread_id}]")
                print(f"   客服: 抱歉，没有找到相关问题的解答，建议您联系人工客服。")
                print("═" * 60 + "\n")
                return
        else:
            hint = f"[路由提示] 用户意图为 {intent}，应该调用 {tool_name} 工具。"

        # ③ Agent invoke（含工具调用 + LLM 包装）
        config = {"configurable": {"thread_id": thread_id}}
        content = f"{hint}\n\n用户问题: {user_query}" if hint else user_query
        result = agent.invoke(
            {"messages": [HumanMessage(content=content)]},
            config=config,
        )

        # ③ 展示智能体决策细节：LLM 调了哪个 tool，工具返回什么
        print(f"\n③ 智能体决策")
        # 找出本轮 LLM 触发的 tool_calls 和 ToolMessage
        new_msgs = result["messages"]
        # 倒序找最近的 tool_call AIMessage 和它对应的 ToolMessage
        recent_tool_calls = []
        for m in new_msgs[::-1]:
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                recent_tool_calls = m.tool_calls
                break
        recent_tool_msgs = [m for m in new_msgs if isinstance(m, ToolMessage)]
        # 用本轮新增的 ToolMessage（取最后 N 条对应 tool_calls 数量）
        relevant_tool_msgs = recent_tool_msgs[-len(recent_tool_calls):] if recent_tool_calls else []

        if recent_tool_calls:
            for i, tc in enumerate(recent_tool_calls):
                args_preview = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                print(f"   LLM 调用工具: {tc['name']}({args_preview})")
                if i < len(relevant_tool_msgs):
                    tm = relevant_tool_msgs[i]
                    content_preview = tm.content[:200].replace("\n", " ")
                    print(f"   工具返回: {content_preview}{'...' if len(tm.content) > 200 else ''}")
        else:
            print(f"   LLM 未调工具（基于历史上下文直接回复）")

        # ④ 结果输出
        last = result["messages"][-1]
        print(f"\n④ 结果输出  [{thread_id}]")
        print(f"   客服: {last.content}")
        print("═" * 60 + "\n")


    # ============ 会话 A：测连贯多轮（"它"指代上文订单号） ============
    print("--- 会话 A（user_001）：测上下文记忆 ---")
    chat("我的订单 JD20240610001 到哪了？", thread_id="user_001")
    chat("它最新的物流呢？", thread_id="user_001")  # ← "它" 应该指 JD20240610001
    chat("好的，那这款商品有保修吗？", thread_id="user_001")  # ← "这款商品" 应该指订单里的耳机

    # ============ 会话 B：另一个用户（thread 隔离验证）============
    print("--- 会话 B（user_002）：thread 隔离 ---")
    chat("推荐一款游戏本", thread_id="user_002")
    chat("便宜点的呢？", thread_id="user_002")  # ← Agent 应该记住"游戏本"语境

    # ============ 会话 C：转人工 → n8n webhook → 邮件（B 线端到端验证）============
    print("--- 会话 C（user_complaint）：转人工创建工单 + n8n 邮件 ---")
    chat("我的订单 JD20240610003 雷蛇耳机有问题", thread_id="user_complaint")
    chat("我要投诉！找客服三次都没解决，要求经理出面赔偿！", thread_id="user_complaint")

