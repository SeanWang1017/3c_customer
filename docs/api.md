# 3C 电商智能客服 Agent 系统 — API/工具说明

> 本文档为对外技术说明，详细描述系统提供的 API 接口和工具函数。

---

## 1. 概述

本系统提供以下核心 API 和工具：

| 类别 | 名称 | 功能 |
|:-----|:-----|:-----|
| Web API | `/` | 主页面 |
| Web API | `/chat` | 对话接口 (SSE) |
| 工具函数 | `query_order()` | 订单查询 |
| 工具函数 | `search_faq()` | FAQ 检索 |
| 工具函数 | `search_product()` | 商品检索 |
| 工具函数 | `create_ticket()` | 工单创建 |

---

## 2. Web API

### 2.1 主页面

```
GET /
```

**描述**：返回客服工作台前端页面

**响应**：HTML 页面

---

### 2.2 对话接口

```
POST /chat
Content-Type: application/json
```

**描述**：处理用户对话，返回 SSE 流式响应（4 阶段）

**请求体**：

```json
{
  "query": "帮我查一下订单 JD20240610001",
  "thread_id": "user_001"
}
```

| 字段 | 类型 | 必填 | 说明 |
|:-----|:-----|:----:|:-----|
| query | string | ✅ | 用户输入的对话内容 |
| thread_id | string | ✅ | 会话 ID，用于隔离多轮对话历史 |

**响应格式**（SSE 流）：

```text
data: {"stage": "user_input", "content": "帮我查一下订单 JD20240610001"}

data: {"stage": "intent_recognition", "content": "意图：order_query (置信度: 0.892)"}

data: {"stage": "agent_decision", "content": "调用工具 query_order(order_id='JD20240610001')..."}

data: {"stage": "result_output", "content": "订单 JD20240610001 状态为已发货..."}
```

**SSE 阶段说明**：

| 阶段 | 说明 |
|:-----|:-----|
| `user_input` | 用户输入回显 |
| `intent_recognition` | 意图识别结果（含 4 重保险信息） |
| `agent_decision` | Agent 决策过程（工具调用、参数提取） |
| `result_output` | 最终回复结果 |

---

## 3. 工具函数

### 3.1 订单查询工具

```python
from src.tools.order_query import query_order

result = query_order(order_id="JD20240610001")
```

**功能**：查询订单状态，从 `data/orders/orders.json` 读取 mock 数据

**参数**：

| 参数 | 类型 | 必填 | 说明 |
|:-----|:-----|:----:|:-----|
| order_id | string | ✅ | 订单号（任意字符串，自动 strip + upper） |

**返回值**：

```python
# 成功
{
    "found": True,
    "order_id": "JD20240610001",
    "status": "已发货",
    "product_name": "雷蛇旋风黑鲨V2耳机",
    "price": 599,
    "quantity": 1,
    "order_time": "2024-06-10 14:30:00",
    "shipping_address": "北京市朝阳区xxx",
    "logistics": {
        "company": "顺丰速运",
        "tracking_no": "SF1234567890",
        "latest_update": "已发货，预计明天送达",
        "updates": [...]
    }
}

# 失败
{
    "found": False,
    "message": "订单 JD20240610001 不存在"
}
```

**错误处理**：
- 订单号为空 → `{"found": False, "message": "订单号不能为空"}`
- 订单不存在 → `{"found": False, "message": "订单 XXX 不存在"}`
- **不抛异常**，LLM 友好

---

### 3.2 FAQ 检索工具

```python
from src.tools.langchain_tools import search_faq

result = search_faq(query="耳机拆封了还能退吗")
```

**功能**：从 FAQ 向量库中检索相似政策文档

**参数**：

| 参数 | 类型 | 必填 | 说明 |
|:-----|:-----|:----:|:-----|
| query | string | ✅ | 查询文本 |

**返回值**：

```python
[
    {
        "content": "商品支持七天无理由退货吗？...部分商品（如已拆封的耳机、内裤等）不支持。",
        "score": 0.82,
        "source": "faq"
    },
    ...
]
```

**实现细节**：
- 调用 `scripts/rag_query.py` 子进程（避免 langchain 进程 segfault）
- BGE-small-zh-v1.5 编码 + FAISS 内积检索
- 返回 top-3 相似文档

---

### 3.3 商品检索工具

```python
from src.tools.langchain_tools import search_product

result = search_product(query="3000元以内 游戏手机")
```

**功能**：从商品向量库中检索相似商品

**参数**：

| 参数 | 类型 | 必填 | 说明 |
|:-----|:-----|:----:|:-----|
| query | string | ✅ | 查询文本 |

**返回值**：

```python
[
    {
        "content": "iQOO 15 系列（12+256GB，¥3999）...",
        "score": 0.78,
        "source": "product"
    },
    ...
]
```

**实现细节**：
- 调用 `scripts/rag_query.py` 子进程
- BGE-small-zh-v1.5 编码 + FAISS 内积检索
- 返回 top-3 相似商品

---

### 3.4 工单创建工具

```python
from src.tools.ticket import create_ticket

result = create_ticket(
    user_id="user_001",
    user_query="我要投诉！耳机有问题",
    conversation_history="[14:30] 用户：我的耳机...\n[14:31] 机器人：...",
    order_id="JD20240610001"
)
```

**功能**：创建工单并通知 n8n（静默失败）

**参数**：

| 参数 | 类型 | 必填 | 说明 |
|:-----|:-----|:----:|:-----|
| user_id | string | ✅ | 用户 ID（来自 Agent 会话 thread_id） |
| user_query | string | ✅ | 用户最新一句完整诉求 |
| conversation_history | string | ❌ | 已格式化的对话历史（多行 `[HH:MM] 角色：内容`） |
| order_id | string | ❌ | 关联订单号（上下文有就带） |

**返回值**：

```python
# 成功
{
    "found": True,
    "ticket_id": "WT20260621163245001",
    "status": "待处理",
    "notified": True,  # n8n webhook 是否成功
    "message": "工单 WT20260621163245001 已创建，已通知人工客服"
}

# 失败（webhook 未配置）
{
    "found": True,
    "ticket_id": "WT20260621163245001",
    "status": "待处理",
    "notified": False,
    "message": "工单 WT20260621163245001 已创建，本地已记录（webhook 未送达）"
}
```

**工单号生成规则**：`WT` + `YYYYMMDDHHMMSS` + 3 位随机数

**n8n Webhook Payload**：

```json
{
    "ticket_id": "WT20260621163245001",
    "user_id": "user_001",
    "order_id": "JD20240610001",
    "user_query": "我要投诉！耳机有问题",
    "conversation_history": "[14:30] 用户：...\n[14:31] 机器人：...",
    "created_at": "2026-06-21T16:32:45"
}
```

**错误处理**：
- 工单落库失败 → `{"found": False, "message": "工单落库失败：..."}`
- n8n webhook 失败 → `notified=False`，不影响主流程
- **不抛异常**，LLM 友好

---

## 4. 意图识别 API

### 4.1 意图分类

```python
from src.intent.classifier import classify_with_confidence

intent, confidence, raw_output = classify_with_confidence("帮我查一下订单")
```

**功能**：本地 LoRA 模型意图分类

**参数**：

| 参数 | 类型 | 必填 | 说明 |
|:-----|:-----|:----:|:-----|
| user_query | string | ✅ | 用户输入文本 |

**返回值**：

```python
(intent, confidence, raw_output)
# 例如: ("order_query", 0.892, "order_query")
```

**意图类别**：

| 意图 | 说明 |
|:-----|:-----|
| order_query | 订单/物流查询 |
| product_intro | 商品咨询/介绍 |
| product_recommend | 商品推荐 |
| policy_qa | 售后政策问答 |
| ticket_transfer | 转人工/投诉 |

---

### 4.2 规则后处理

```python
from src.intent.postprocess import postprocess

intent, was_changed, reason = postprocess("这款耳机保修多久", "product_intro", 0.8)
```

**功能**：关键词规则后处理（第 2 重保险）

**参数**：

| 参数 | 类型 | 必填 | 说明 |
|:-----|:-----|:----:|:-----|
| user_query | string | ✅ | 用户输入文本 |
| intent | string | ✅ | 初始意图分类结果 |
| confidence | float | ✅ | 置信度 |

**返回值**：

```python
(intent, was_changed, reason)
# 例如: ("policy_qa", True, "通用售后政策规则触发")
```

**规则示例**：

| 关键词 | 改判 |
|:-------|:-----|
| "这[款台个](.{0,8})?(保修\|参数\|规格...)" | product_intro |
| "(7天\|三包\|发票\|国行)..." | policy_qa |
| "投诉/转人工/经理/差评" | ticket_transfer |

---

## 5. 意图路由 API

### 5.1 意图路由

```python
from src.agent.intent_router import classify_and_route

result = classify_and_route("帮我查一下订单", llm=None)
```

**功能**：4 重保险意图识别 + 路由

**参数**：

| 参数 | 类型 | 必填 | 说明 |
|:-----|:-----|:----:|:-----|
| user_query | string | ✅ | 用户输入文本 |
| llm | BaseChatModel | ❌ | 云端 LLM（传了启用第 3 重保险） |

**返回值**：

```python
{
    "intent": "order_query",
    "confidence": 0.892,
    "tool": "query_order",
    "was_changed": False,
    "reason": ""
}
```

---

## 6. 数据格式

### 6.1 订单数据格式

文件位置：`data/orders/orders.json`

```json
{
    "order_id": "JD20240610001",
    "status": "已发货",
    "product_name": "雷蛇旋风黑鲨V2耳机",
    "price": 599,
    "quantity": 1,
    "order_time": "2024-06-10 14:30:00",
    "shipping_address": "北京市朝阳区xxx",
    "logistics": {
        "company": "顺丰速运",
        "tracking_no": "SF1234567890",
        "updates": [
            {
                "time": "2024-06-10 15:00:00",
                "desc": "已发货，预计明天送达"
            }
        ]
    }
}
```

### 6.2 工单数据格式

文件位置：`data/orders/tickets.json`

```json
{
    "ticket_id": "WT20260621163245001",
    "order_id": "JD20240610001",
    "user_id": "user_001",
    "reason": "我要投诉！耳机有问题",
    "status": "待处理",
    "created_at": "2026-06-21T16:32:45",
    "assigned_to": ""
}
```

### 6.3 FAQ 数据格式

文件位置：`data/faq/faq.json`

```json
{
    "id": "FQ001",
    "question": "商品支持七天无理由退货吗？",
    "answer": "部分商品（如已拆封的耳机、内裤等）不支持。",
    "category": "退货政策"
}
```

---

## 7. 环境变量

项目通过 `.env` 文件配置环境变量：

| 变量名 | 必填 | 说明 |
|:-------|:----:|:-----|
| `DASHSCOPE_API_KEY` | ✅ | 阿里云百炼 API Key |
| `DASHSCOPE_BASE_URL` | ✅ | API 地址（`https://dashscope.aliyuncs.com/compatible-mode/v1`）|
| `DASHSCOPE_MODEL` | ✅ | 模型名称（`qwen-flash`）|
| `N8N_WEBHOOK_URL` | ❌ | n8n webhook URL（不配置则仅本地落库）|

---

## 8. 错误处理

### 8.1 设计原则

- **LLM 友好**：工具函数返回结构化错误信息，不抛异常
- **静默失败**：n8n webhook 失败不影响主流程
- **本地优先**：工单落库永远优先成功

### 8.2 错误响应格式

```python
# 标准错误格式
{
    "found": False,
    "message": "错误描述"
}

# 工单创建失败
{
    "found": False,
    "message": "工单落库失败：FileNotFoundError: ..."
}
```

---

## 9. 使用示例

### 9.1 CLI 模式

```python
from src.agent.customer_service_agent import chat

# 单轮对话
response = chat("帮我查一下订单 JD20240610001", thread_id="user_001")
print(response)

# 多轮对话
chat("这款耳机降噪怎么样？", thread_id="user_001")
chat("它多少钱？", thread_id="user_001")  # "它" = 上一轮的耳机
```

### 9.2 Web 模式

```bash
# 启动 Web 服务
python src/web/run.py

# 浏览器访问
# http://localhost:8089
```

### 9.3 工具函数调用

```python
# 订单查询
from src.tools.order_query import query_order
result = query_order("JD20240610001")

# FAQ 检索
from src.tools.langchain_tools import search_faq
results = search_faq("耳机拆封了还能退吗")

# 商品检索
from src.tools.langchain_tools import search_product
results = search_product("3000元以内 游戏手机")

# 工单创建
from src.tools.ticket import create_ticket
result = create_ticket(
    user_id="user_001",
    user_query="我要投诉",
    order_id="JD20240610001"
)
```

---

## 10. 依赖说明

| 依赖 | 版本 | 用途 |
|:-----|:-----|:-----|
| fastapi | - | Web 框架 |
| uvicorn | - | ASGI 服务器 |
| langchain | 1.x | Agent 框架 |
| langgraph | - | 状态管理 |
| faiss | - | 向量检索 |
| sentence-transformers | - | BGE 编码 |
| requests | - | HTTP 请求（n8n webhook） |
| python-dotenv | - | 环境变量加载 |
