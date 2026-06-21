"""
Step 2: 构造可检索语料
- 把 load_data 返回的 dict 列表，转成 (texts, metadatas)
- 不嵌入、不切分 — 只做"文本构造"
- texts 给 BGE 编码；metadatas 检索后回显
"""

from load_data import load_products, load_faq

def build_faq_corpus(faq_list: list[dict]) -> tuple[list[str], list[dict]]:
    """构造 FAQ 可检索语料

    文本 = 问题 + "\n" + 答案（让 query 和 doc 语义空间更接近）
    metadata = 原始 dict（含 id/category/question/answer，检索后回显）

    Returns:
        (texts, metadatas)
        - texts: 30 条可检索文本
        - metadatas: 30 条原始 dict
    """
    texts = []
    metadatas = []
    for item in faq_list:
        texts.append(f"{item['question']}\n{item['answer']}")
        metadatas.append(item)
    return texts, metadatas

def build_products_corpus(products_list: list[dict]) -> tuple[list[str], list[dict]]:
    """构造商品可检索语料

    文本策略：拼接 title + 4 个关键属性（category/price/sales/rating）
    metadata：完整 dict（检索后展示所有 9 个字段）

    拼接示例：
    "title: 雷蛇战锤狂鲨V3 | category: Earphone | price: 339.0
     | sales: 100+人复购 | rating: 99%好评"

    Returns:
        (texts, metadatas)
        - texts: 3929 条可检索文本
        - metadatas: 3929 条原始 dict
    """

    texts = []
    metadatas = []
    for p in products_list:
        parts = [
            f"title: {p.get('title', '')}",
            f"category: {p.get('category', '')}",
            f"price: {p.get('price', '')}",
            f"sales: {p.get('sales', '')}",
            f"rating: {p.get('rating', '')}",
        ]
        text = " | ".join(parts)
        texts.append(text)
        metadatas.append(p)
    return texts, metadatas

if __name__ == "__main__":
    # 验收：跑这个文件能验证 Step 2 的文本构造逻辑
    faq_list = load_faq()
    faq_texts, faq_metas = build_faq_corpus(faq_list)
    print(f"FAQ texts: {len(faq_texts)} 条")
    print(f"  样本 0 text: {faq_texts[0]!r}")
    print(f"  样本 0 question: {faq_metas[0]['question']}")

    products_list = load_products()
    prod_texts, prod_metas = build_products_corpus(products_list)
    print(f"\n商品 texts: {len(prod_texts)} 条")
    print(f"  样本 0 text: {prod_texts[0]!r}")
    print(f"  样本 0 title: {prod_metas[0].get('title', '?')}")
