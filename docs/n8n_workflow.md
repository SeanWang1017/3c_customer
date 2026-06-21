# n8n 工单通知工作流 — 集成方案

> 日期：2026-06-21（S8.3 完成）
> n8n 版本：2.2.6（自部署 Docker）
> SMTP：163 邮箱发件 → QQ 邮箱收件，跨服务商投递验证 OK

---

## 1. 用途

当用户在对话中触发 **转人工** 意图（投诉/差评/经理/赔偿 等关键词），项目代码自动：

1. 生成工单号 `WT + YYYYMMDDHHMMSS + 3位随机`
2. 本地落库 [data/orders/tickets.json](../data/orders/tickets.json)
3. POST 到 n8n webhook
4. n8n 调用 163 SMTP 发送一封**含完整对话历史**的工单邮件给客服邮箱

设计目标：客服打开邮件**直接能介入**处理，不只是"有工单了"的简单通知。

---

## 2. 架构

```text
LangChain Agent
    │
    │ 用户消息走 chat() 函数
    │
    ▼
src/agent/intent_router.py
    ├─ 第 1 重保险：本地 LoRA 89.8%
    └─ 第 2 重保险：关键词规则
            │
            │ intent == "ticket_transfer"
            ▼
src/agent/customer_service_agent.py  chat() 的 ticket_transfer 分支
    │ ① agent.get_state() 提取历史
    │ ② format_history_for_ticket() 整理多行可读文本
    │ ③ extract_order_id_from_history() 正则抓订单号
    ▼
src/tools/ticket.py  create_ticket()
    ├─ generate_ticket_id() → WT20260621...
    ├─ 追加到 data/orders/tickets.json（必成功）
    └─ requests.post(N8N_WEBHOOK_URL, json=payload, timeout=3)  ← 静默失败
            │
            ▼
n8n Webhook 节点（POST /webhook/ticket-created）
            │
            ▼
n8n Edit Fields 节点（拼 subject + body）
            │
            ▼
n8n Send Email 节点（163 SMTP smtp.163.com:465 SSL）
            │
            ▼
QQ 邮箱（或任何外部邮箱）
```

---

## 3. n8n 工作流 3 节点配置

### 3.1 Webhook 节点

| 字段 | 值 |
|:----|:----|
| HTTP Method | `POST` |
| Path | `ticket-created` |
| Authentication | None |
| Respond | Immediately |

生产 URL：`http://localhost:5678/webhook/ticket-created`
测试 URL：`http://localhost:5678/webhook-test/ticket-created`（需先点 Listen for test event）

### 3.2 Edit Fields (Set) 节点

两个字段（全部切换为 Expression 模式）：

**subject**：
```text
[3C客服-转人工]：{{ $json.body.ticket_id }}\n用户：{{ $json.body.user_id }}
```

**body**：
```text
您好，\n\n以下是新转人工的工单详情，请您介入处理：\n工单号：{{ $json.body.ticket_id }}\n用户ID：{{ $json.body.user_id }}\n关联订单：{{ $json.body.order_id || "（用户未提及）" }}\n创建时间：{{ $json.body.created_at }}\n【用户最新诉求】：{{ $json.body.user_query }}\n【对话历史】：{{ $json.body.conversation_history }}\n请登录客服工作台跟进！\n\n3C 电商智能客服 Agent 系统
```

### 3.3 Send Email 节点

| 字段 | 值 |
|:----|:----|
| From Email | `<请填写发件邮箱，如 your_sender@163.com>` |
| To Email | （收件邮箱，可填客服邮箱） |
| Subject | `{{ $json.subject }}` |
| Email Format | Text |
| Text | `{{ $json.body }}` |

**SMTP credential** (`163-smtp`)：

| 字段 | 值 |
|:----|:----|
| Host | `smtp.163.com` |
| Port | `465` |
| SSL/TLS | **ON** |
| User | `<163 邮箱完整地址>` |
| Password | `<163 SMTP 授权码，非登录密码>` |

> ⚠️ 163 邮箱需先在 **设置 → POP3/SMTP/IMAP** 开启 SMTP 服务并生成专用授权码，**不能用登录密码**。

---

## 4. Webhook Payload 格式（项目代码 POST）

```json
{
  "ticket_id": "WT20260621165316539",
  "user_id": "user_complaint",
  "order_id": "JD20240610003",
  "user_query": "我要投诉！找客服三次都没解决，要求经理出面赔偿！",
  "conversation_history": "[16:53] 用户：我的订单 JD20240610003 雷蛇耳机有问题\n[16:53] 机器人：您的订单 JD20240610003 已发货...\n[16:53] 用户：我要投诉！...",
  "created_at": "2026-06-21T16:53:16"
}
```

字段来源：

| 字段 | 项目代码生成位置 |
|:----|:----|
| `ticket_id` | [src/tools/ticket.py](../src/tools/ticket.py) `generate_ticket_id()` |
| `user_id` | LangGraph 会话 `thread_id`（CLI 默认 `user_cli`，Web 用 session id） |
| `order_id` | [src/agent/customer_service_agent.py](../src/agent/customer_service_agent.py) `extract_order_id_from_history()` 正则 `JD\d{6,}` |
| `user_query` | 用户本轮原始消息 |
| `conversation_history` | `format_history_for_ticket()` 从 LangGraph state 提取 |
| `created_at` | `datetime.now().isoformat(timespec="seconds")` |

**已删除字段**：

| 字段 | 删除原因 |
|:----|:----|
| ~~`priority`~~ | 项目无优先级判定逻辑，硬编码 `high` 是演 |
| ~~`summary`~~ | LLM 摘要会丢信息，改为完整 `user_query` |

---

## 5. 故障与降级策略

| 故障 | 行为 |
|:----|:----|
| `.env` 未配置 `N8N_WEBHOOK_URL` | `_post_to_n8n` 直接返回 False，跳过 POST |
| n8n 容器宕机 | requests timeout 3s，print warning 到 stderr，工单仍落库 |
| n8n webhook 返回非 2xx | print 状态码到 stderr，工单仍落库 |
| 163 SMTP 拒绝（授权码失效等） | n8n 那边 Send Email 节点失败，但 webhook 已 200 ACK，对项目代码透明 |

**核心原则**：webhook 是"额外动作"，**永远不影响工单本地落库**。

---

## 6. 部署适配

| 部署方式 | `N8N_WEBHOOK_URL` 取值 |
|:----|:----|
| 本机项目代码 + 本机 n8n docker | `http://localhost:5678/webhook/ticket-created` |
| 远程服务器 | `http://<服务器IP>:5678/webhook/ticket-created` |

> 本地部署时使用 `localhost`；如需远程访问，请替换为实际服务器地址。

---

## 7. 验证步骤

### 7.1 单独验证 ticket 工具

```bash
python src/tools/ticket.py
```

预期输出：`ticket_id` 生成、`notified=True`、QQ 邮箱收到邮件。

### 7.2 端到端验证（Agent 触发）

```bash
python test/smoke_b_ticket_e2e.py
```

预期：2 轮对话场景，第 2 轮投诉触发 ticket_transfer → 创建工单 → 邮件含完整历史 + 关联订单。

---

## 8. 实测投递记录

| 时间 | ticket_id | 投递方向 | 状态 |
|:----|:----|:----|:----:|
| 2026-06-21 16:46:22 | WT20260621164622844 | 163 → QQ | ✅ 收件箱 |
| 2026-06-21 16:53:16 | WT20260621165316539 | 163 → QQ | ✅ 收件箱（含上下文 + 订单号） |

跨服务商投递验证通过，反垃圾邮件未拦截。
