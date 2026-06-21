"""
3C 智能客服工作台 - FastAPI 后端
- POST /chat：SSE 流式返回 4 阶段决策详情
- GET /：返回单文件前端 index.html
- 启动：python src/web/run.py
"""
import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from src.agent.customer_service_agent import (
    build_agent,
    build_llm,
    extract_order_id_from_history,
    format_history_for_ticket,
)
from src.agent.intent_router import classify_and_route
from src.tools.ticket import create_ticket

# 全局 LLM + Agent（lifespan 启动时初始化）
_llm = None
_agent = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """启动时初始化 LLM 和 Agent（避免每次请求都重建）。"""
    global _llm, _agent
    _llm = build_llm()
    _agent = build_agent(llm=_llm)
    yield


app = FastAPI(title="3C 智能客服工作台", lifespan=lifespan)


class ChatRequest(BaseModel):
    query: str
    thread_id: str = "default"


async def web_chat(
    user_query: str, thread_id: str
) -> AsyncIterator[dict]:
    """Web 版 chat：4 阶段 SSE 推送。

    阶段：
    ① input        - 用户输入
    ② intent       - 意图识别（第 1 重 + 第 2 重 + 可选第 3 重）
    ③ tool_call    - LLM 调了哪个工具 + 参数
    ③ tool_result  - 工具返回的数据预览
    ④ reply        - 最终客服回复
    ④ review_required - 触发第 4 重保险（人工审核）
    done           - 结束
    """
    # ① 用户输入
    yield {
        "stage": "input",
        "user": user_query,
        "thread_id": thread_id,
    }

    # ② 意图识别（第 1 重 + 第 2 重 + 可选第 3 重）—— 同步调用，放到线程池避免阻塞事件循环
    route = await asyncio.to_thread(
        classify_and_route, user_query, _llm
    )
    intent = route["intent"]
    tool_name = route["tool"]
    yield {
        "stage": "intent",
        "intent": intent,
        "conf": route["confidence"],
        "tool": tool_name or "无（直接回复）",
        "was_changed": route.get("was_changed", False),
        "reason": route.get("reason", ""),
    }

    # 路由结果分类处理
    if tool_name is None:
        if intent == "ticket_transfer":
            # ── 转人工：跳过 LLM，直接调 create_ticket（落库 + n8n 通知 → 邮件）
            tt_config = {"configurable": {"thread_id": thread_id}}
            try:
                state = _agent.get_state(tt_config)
                prior_messages = (
                    state.values.get("messages", []) if state and state.values else []
                )
            except Exception as e:
                # 历史拿不到不阻塞工单创建，但留痕（checkpointer 异常 / thread_id 不存在时方便排查）
                print(f"[web] get_state failed: {type(e).__name__}: {e}", file=sys.stderr)
                prior_messages = []

            all_msgs_for_ticket = prior_messages + [HumanMessage(content=user_query)]
            history_text = format_history_for_ticket(all_msgs_for_ticket)
            order_id_in_history = extract_order_id_from_history(all_msgs_for_ticket)

            ticket_result = await asyncio.to_thread(
                create_ticket,
                user_id=thread_id,
                user_query=user_query,
                conversation_history=history_text,
                order_id=order_id_in_history,
            )

            tid = ticket_result.get("ticket_id", "?")
            notified = ticket_result.get("notified", False)
            reply_text = (
                f"已为您创建工单 {tid}，并通知人工客服跟进，请稍候。"
                if notified
                else f"已为您创建工单 {tid}（系统已记录，工单通知服务暂不可达，仍可由客服查询本地工单）。"
            )
            yield {
                "stage": "decision",
                "summary": "转人工（直调 create_ticket，跳过 LLM）",
                "ticket_id": tid,
                "notified": notified,
                "order_id": order_id_in_history,
            }
            yield {
                "stage": "reply",
                "content": reply_text,
            }
            yield {"stage": "review_required"}
            yield {"stage": "done"}
            return
        elif intent is None:
            # 路由失败 → LLM 看历史自主决定
            hint = ""
        else:
            yield {
                "stage": "reply",
                "content": "抱歉，没有找到相关问题的解答，建议您联系人工客服。",
            }
            yield {"stage": "review_required"}
            yield {"stage": "done"}
            return
    else:
        hint = f"[路由提示] 用户意图为 {intent}，应该调用 {tool_name} 工具。"

    # ③ Agent invoke（拿到工具调用 + 回复）
    config = {"configurable": {"thread_id": thread_id}}
    content = (
        f"{hint}\n\n用户问题: {user_query}" if hint else user_query
    )
    result = await asyncio.to_thread(
        _agent.invoke,
        {"messages": [HumanMessage(content=content)]},
        config=config,
    )

    # ③ 找出本轮 LLM 触发的 tool_calls 和 ToolMessage
    new_msgs = result["messages"]
    recent_tool_calls = []
    for m in new_msgs[::-1]:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            recent_tool_calls = m.tool_calls
            break
    recent_tool_msgs = [
        m for m in new_msgs if isinstance(m, ToolMessage)
    ]
    relevant_tool_msgs = (
        recent_tool_msgs[-len(recent_tool_calls):]
        if recent_tool_calls
        else []
    )

    if recent_tool_calls:
        for i, tc in enumerate(recent_tool_calls):
            yield {
                "stage": "tool_call",
                "tool": tc["name"],
                "args": tc["args"],
            }
            if i < len(relevant_tool_msgs):
                tm = relevant_tool_msgs[i]
                yield {
                    "stage": "tool_result",
                    "content": tm.content[:300],  # 预览 300 字符
                }
    else:
        # LLM 没调工具（基于历史上下文自主回复）
        yield {
            "stage": "tool_call",
            "tool": "无（LLM 自主回复）",
            "args": {},
        }

    # ④ 结果输出
    last = result["messages"][-1]
    yield {"stage": "reply", "content": last.content}
    yield {"stage": "review_required"}

    yield {"stage": "done"}


@app.post("/chat")
async def chat(req: ChatRequest):
    """SSE 流式聊天接口。"""

    async def event_generator():
        try:
            async for event in web_chat(req.query, req.thread_id):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            err = {"stage": "error", "message": str(e)}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁用 nginx buffering
        },
    )


@app.get("/")
async def root():
    """返回前端单文件 index.html。"""
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    """健康检查。"""
    return {
        "status": "ok",
        "llm_loaded": _llm is not None,
        "agent_loaded": _agent is not None,
    }
