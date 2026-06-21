"""
工单创建工具 - S5 Agent 工具
- 生成 ticket_id（WT + YYYYMMDDHHMMSS + 3 位随机数）
- 工单落库 data/orders/tickets.json
- 同步 POST 到 n8n webhook（静默失败，不影响落库）

设计原则：
- POST 失败不抛异常、不影响 LLM 工具调用结果
- 工单落库永远优先成功，webhook 通知是"额外动作"
"""
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

TICKETS_PATH = PROJECT_ROOT / "data" / "orders" / "tickets.json"
WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "").strip()
WEBHOOK_TIMEOUT_SEC = 3


def generate_ticket_id() -> str:
    """生成工单号：WT + YYYYMMDDHHMMSS + 3 位随机数。

    Example:
        WT20260621163245001
    """
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    suffix = f"{random.randint(0, 999):03d}"
    return f"WT{ts}{suffix}"


def _post_to_n8n(payload: dict) -> bool:
    """同步 POST 工单到 n8n webhook，静默失败。

    Returns:
        True: webhook 配置且 POST 2xx；False: 未配置/超时/失败
    """
    if not WEBHOOK_URL:
        return False
    try:
        resp = requests.post(
            WEBHOOK_URL,
            json=payload,
            timeout=WEBHOOK_TIMEOUT_SEC,
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        if 200 <= resp.status_code < 300:
            return True
        print(
            f"[ticket] n8n webhook 返回 {resp.status_code}（已忽略，工单已本地落库）",
            file=sys.stderr,
        )
        return False
    except requests.exceptions.RequestException as e:
        print(
            f"[ticket] n8n webhook 调用失败 {type(e).__name__}（已忽略，工单已本地落库）",
            file=sys.stderr,
        )
        return False


def _append_to_tickets_json(ticket: dict) -> None:
    """追加工单到 data/orders/tickets.json（保持原有数据结构）。"""
    if TICKETS_PATH.exists():
        with open(TICKETS_PATH, "r", encoding="utf-8") as f:
            tickets = json.load(f)
    else:
        tickets = []
    tickets.append(ticket)
    with open(TICKETS_PATH, "w", encoding="utf-8") as f:
        json.dump(tickets, f, ensure_ascii=False, indent=2)


def create_ticket(
    user_id: str,
    user_query: str,
    conversation_history: str = "",
    order_id: str = "",
) -> dict:
    """创建工单并通知 n8n。

    Args:
        user_id: 用户ID（来自 Agent 会话 thread_id，CLI 模式用 "user_cli"）
        user_query: 用户最新一句完整诉求（不做摘要）
        conversation_history: 已格式化的对话历史，多行 [HH:MM] 角色：内容；可空
        order_id: 关联订单号（上下文有就带，没有就空串）

    Returns:
        {
            "found": True,
            "ticket_id": "WT...",
            "status": "待处理",
            "notified": True/False,  # n8n webhook 是否成功
            "message": "工单已创建..."
        }
        不抛异常（LLM 友好），失败也只在 message 里说明。
    """
    ticket_id = generate_ticket_id()
    created_at = datetime.now().isoformat(timespec="seconds")

    # 本地落库结构（兼容现有 tickets.json 字段：ticket_id/order_id/user_id/reason/status/created_at/assigned_to）
    ticket_record = {
        "ticket_id": ticket_id,
        "order_id": order_id,
        "user_id": user_id,
        "reason": user_query,
        "status": "待处理",
        "created_at": created_at,
        "assigned_to": "",
    }

    try:
        _append_to_tickets_json(ticket_record)
    except Exception as e:
        return {
            "found": False,
            "message": f"工单落库失败：{type(e).__name__}: {e}",
        }

    # n8n webhook payload（独立结构，包含完整诊断信息）
    webhook_payload = {
        "ticket_id": ticket_id,
        "user_id": user_id,
        "order_id": order_id,
        "user_query": user_query,
        "conversation_history": conversation_history or "（无历史对话）",
        "created_at": created_at,
    }
    notified = _post_to_n8n(webhook_payload)

    return {
        "found": True,
        "ticket_id": ticket_id,
        "status": "待处理",
        "notified": notified,
        "message": (
            f"工单 {ticket_id} 已创建，"
            + ("已通知人工客服" if notified else "本地已记录（webhook 未送达）")
        ),
    }


if __name__ == "__main__":
    # 冒烟测试
    print("=== 工单创建工具冒烟测试 ===\n")

    print(f"WEBHOOK_URL = {WEBHOOK_URL or '(未配置，跳过 POST)'}\n")

    test_history = "\n".join([
        "[16:30:01] 用户：我的雷蛇耳机左耳没声音",
        "[16:30:05] 机器人：您好，请提供订单号方便我们查询",
        "[16:30:32] 用户：JD20240610003",
        "[16:31:00] 机器人：已为您查到订单，请描述具体问题",
        "[16:31:45] 用户：我要投诉！找过三次都没解决，要求经理赔偿！",
    ])

    result = create_ticket(
        user_id="user_smoke",
        user_query="我要投诉！上次买的雷蛇耳机左耳完全没声音，已经找客服三次都没解决，要求经理出面给我赔偿！",
        conversation_history=test_history,
        order_id="JD20240610003",
    )

    print("结果：")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()
    print(f"→ 检查 163 邮箱是否收到新邮件（subject 含 {result.get('ticket_id', '?')}）")
    print(f"→ 检查 data/orders/tickets.json 是否多了一条记录")
