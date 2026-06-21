"""
订单查询工具 - S5 Agent 工具
- 从 data/orders/orders.json 读取 mock 订单数据
- 提供 query_order(order_id) 函数
- 错误时返回 {found: False, message: ...} 而非抛异常（LLM 友好）
- 订单号格式不固定：查不到就返回"不存在"
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ORDERS_PATH = PROJECT_ROOT / "data" / "orders" / "orders.json"

_orders_cache: list[dict] | None = None


def _load_orders() -> list[dict]:
    """单例加载订单数据（首次调用读文件，后续走内存缓存）"""
    global _orders_cache
    if _orders_cache is None:
        with open(ORDERS_PATH, "r", encoding="utf-8") as f:
            _orders_cache = json.load(f)
    return _orders_cache


def _format_order(order: dict) -> dict:
    """把原始订单转成工具输出格式：空串标准化为 None，添加 latest_update"""
    logistics = order.get("logistics") or {}
    updates = logistics.get("updates") or []
    latest = updates[-1]["desc"] if updates else None

    tracking_no = (logistics.get("tracking_no") or "").strip()

    return {
        "found": True,
        "order_id": order["order_id"],
        "status": order["status"],
        "product_name": order["product_name"],
        "price": order["price"],
        "quantity": order["quantity"],
        "order_time": order["order_time"],
        "shipping_address": order["shipping_address"],
        "logistics": {
            "company": logistics.get("company") or None,
            "tracking_no": tracking_no or None,
            "latest_update": latest,
            "updates": updates,
        },
    }


def query_order(order_id: str) -> dict:
    """
    查询订单状态（Mock 数据，从 data/orders/orders.json 读取）

    Args:
        order_id: 订单号（任意字符串，自动 strip + upper）
                  查不到时返回 found=False

    Returns:
        成功：{"found": True, "order_id": ..., "status": ..., ...}
        失败：{"found": False, "message": "..."}（不抛异常，LLM 友好）
    """
    if not order_id or not isinstance(order_id, str):
        return {"found": False, "message": "订单号不能为空"}

    order_id = order_id.strip().upper()

    for order in _load_orders():
        if order["order_id"] == order_id:
            return _format_order(order)

    return {"found": False, "message": f"订单 {order_id} 不存在"}


if __name__ == "__main__":
    # 冒烟测试
    print("=== 订单查询工具冒烟测试 ===\n")

    test_cases = [
        ("JD20240610001", "已签收完整物流"),
        ("JD20240610009", "待付款无物流"),
        ("JD20240610010", "已完成"),
        ("JD20240610006", "待发货无单号"),
        ("JD99999999999", "不存在"),
        ("invalid", "任意字符串（找不到）"),
        ("", "空字符串"),
        ("  jd20240610001  ", "小写+空格（自动 normalize）"),
    ]

    for tid, desc in test_cases:
        result = query_order(tid)
        print(f"[{desc}] order_id={tid!r}")
        print(f"  → found={result['found']}", end="")
        if result["found"]:
            print(
                f", status={result['status']}, "
                f"latest={result['logistics']['latest_update']}"
            )
        else:
            print(f", message={result['message']}")
        print()