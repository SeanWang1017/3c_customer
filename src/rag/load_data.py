"""
Step 1: 数据加载
- 加载 FAQ (JSON) + 商品 (XLSX) 原始数据
- 不嵌入、不切分、不存库 — 只做加载 + 简单结构化
- 后续 Step 2-6 会用这里返回的 list[dict]
"""

import json
from pathlib import Path

# ⚠️ openpyxl 不能和 sentence_transformers + faiss 同时在内存
# 故用 lazy import：在 load_products() 内部 import
# 适用场景：如果 build_index.py 已经 import 了这两个库，
#         load_data 不会触发 openpyxl 加载
import importlib

# === 路径常量 ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FAQ_PATH = PROJECT_ROOT / "data" / "faq" / "faq.json"
PRODUCTS_PATH = PROJECT_ROOT / "data" / "processed" / "jd_all_cleaned.xlsx"

def load_faq():
    """加载 30条 FAQ 数据"""
    with open(FAQ_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_products():
    """加载 3929 条 3C 商品

    Excel 表头: title, price, sales, rating, store, category,
                raw_price, raw_sales, raw_rating
    返回: [{"title": "雷蛇...", "price": 339.0, ...}, ...]
    """
    openpyxl = importlib.import_module("openpyxl")  # lazy import
    wb = openpyxl.load_workbook(PRODUCTS_PATH, read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))

    # 第 1 行是表头，转字符串
    headers = [str(h) for h in rows[0]]

    products = []
    for r in rows[1:]: # 从第 2 行开始是数据
        # 每行 zip 成 dict，None 转空串
        record = {
            h:("" if v is None else v)
            for h, v in zip(headers, r)
        }
        products.append(record)
    return products

if __name__ == "__main__":
    #验收数据
    faq = load_faq()
    print(f"FAQ: {len(faq)} 条")
    print(f"  样本 0: {faq[0]}")

    products = load_products()
    print(f"\n商品: {len(products)} 条")
    print(f"  样本 0 keys: {list(products[0].keys())}")
    print(f"  样本 0 title: {products[0].get('title', '?')[:50]}")