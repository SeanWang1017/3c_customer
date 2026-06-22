# 3C 电商智能客服 Agent 系统

> **项目状态**：✅ 已完成（2026-06-21）  
> **最终成绩**：意图分类 89.8% + RAG 93.3% + 端到端 91.0/100

面向 3C 电商场景的智能客服系统。
**本地小模型分类意图 + 云端大模型生成回复 + 客服人工审核后发出。**

## 功能

- 商品咨询 / 推荐
- 售后政策问答
- 订单查询（含物流）
- 转人工
- 退换货申请（走转人工流程）

## 技术栈

| 模块 | 技术 |
|:----|:------|
| 本地意图模型 | Qwen2.5-0.5B + QLoRA（5 分类，89.8% 准确率）|
| 云端回复模型 | qwen-flash（阿里云百炼 OpenAI 兼容 API）|
| Agent | LangChain 1.x |
| RAG | FAISS + BGE-small-zh-v1.5 |
| 前端 | FastAPI + SSE |
| 训练 | LLaMA Factory |
| 工单通知 | n8n + 163 SMTP（转人工时自动邮件升级）|

## 项目结构

```text
3C_customer_agent/
├── README.md                # 本文件
├── requirements.txt         # Python 依赖清单
├── .env.example             # 环境变量模板（不含真实 Key）
├── .gitignore
│
├── config/                  # LLaMA Factory 训练配置
│   └── train_lora.yaml      # QLoRA 训练参数（int4, rank=16, lr=3e-5, epoch=3）
│
├── data/                    # 数据目录
│   ├── dataset_info.json    # LLaMA Factory 数据集注册（alpaca 格式）
│   ├── intent_classify_train.jsonl  # 意图分类训练集（2028 条）
│   ├── intent_classify_val.jsonl    # 意图分类验证集（227 条）
│   ├── intent_classify_test.jsonl   # 意图分类测试集（245 条）
│   ├── train.jsonl / val.jsonl / test.jsonl  # 源数据（sharegpt 格式，3482 条）
│   ├── intent/              # 各意图原始语料
│   ├── faq/faq.json         # 售后政策 FAQ（33 条）
│   ├── orders/              # 模拟订单 + 工单数据
│   │   ├── orders.json
│   │   └── tickets.json
│   ├── processed/           # 清洗后商品数据
│   │   └── jd_all_cleaned.xlsx  # 3929 条 3C 商品
│   ├── raw/jddc/            # JDDC 真实客服对话（脱敏）
│   └── vector_store/        # FAISS 向量库
│       ├── faq.index / faq.meta.json
│       └── products.index / products.meta.json
│
├── models/                  # 本地模型（ModelScope 下载）
│   ├── Qwen2.5-0.5B/        # 基座模型（540M 参数，约 942MB）
│   └── embeddings/
│       └── bge-small-zh-v1.5/  # 中文 embedding 模型
│
├── saves/                   # LoRA 训练产物
│   └── qwen2.5-0.5b-intent-lora/  # 意图分类 LoRA 适配器
│       ├── checkpoint-200/
│       └── checkpoint-400/
│
├── scripts/                 # 独立脚本
│   ├── generate_intent_data.py  # 从源数据提取分类样本
│   └── rag_query.py             # RAG 子进程入口（绕开 langchain segfault）
│
├── test/                    # 评测脚本
│   ├── test_intent.py       # 意图分类评测（P/R/F1 + 混淆矩阵）
│   ├── test_rag.py          # RAG 端到端评测
│   └── eval_e2e.py          # 端到端综合评测（5 query × 4 阶段）
│
├── src/                     # 源码
│   ├── intent/              # 意图识别
│   │   ├── classifier.py    # 第 1 重保险：本地 LoRA 加载 + 置信度
│   │   └── postprocess.py   # 第 2 重保险：关键词规则后处理
│   ├── rag/                 # RAG 检索
│   │   ├── load_data.py     # xlsx / json 加载
│   │   ├── build_corpus.py  # 构建语料
│   │   ├── embedder.py      # BGE 编码（子进程调用）
│   │   ├── build_index.py   # FAISS 建索引
│   │   └── retriever.py     # 双库检索（FAQ + 商品）
│   ├── tools/               # LangChain 工具
│   │   ├── order_query.py   # 订单查询
│   │   ├── ticket.py        # 工单创建 + n8n webhook 通知
│   │   └── langchain_tools.py  # 3 个 @tool 封装
│   ├── agent/               # 智能体
│   │   ├── customer_service_agent.py  # 主 Agent（多轮 + 4 阶段可视化）
│   │   ├── intent_router.py           # 4 重保险意图路由（本地→规则→云端→人工）
│   │   └── system_prompt.txt          # 4 段系统 prompt
│   └── web/                 # FastAPI 客服工作台
│       ├── app.py           # SSE 4 阶段流式
│       ├── run.py           # 启动脚本（端口 8089）
│       └── static/index.html  # 双栏 + 4 阶段决策窗 + 第 4 重保险（人工审核）
│
├── reports/                 # 评测报告（JSON）
│
├── docs/                    # 技术文档
│   ├── architecture.md      # 系统架构说明（含架构图、核心算法）
│   ├── api.md               # API/工具说明
│   ├── evaluation.md        # 综合评测报告
│   ├── mindmap_simple.md    # 项目思维导图
│   ├── n8n_workflow.md      # n8n 工单通知集成说明
│   ├── n8n_workflow.json    # n8n 工作流导出（脱敏占位符版）
│   └── data_generation_guide.md  # 对话数据生成指南
│
└── 3C电商智能客服Agent系统-答辩.pptx  # 答辩 PPT
```

## 快速开始

### 环境要求

- Python 3.10+
- 8GB 显存的 NVIDIA 显卡（仅训练时需要；推理/启动 Web 不需要 GPU）
- conda（推荐 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)）

### 1. 创建 conda 环境并安装依赖

```bash
# 创建并激活 conda 环境（Python 3.10）
conda create -n graph python=3.10 -y
conda activate graph

# 安装 Python 依赖
pip install -r requirements.txt

# 训练用额外依赖（不需要训练可跳过）
pip install llamafactory
```

> 如果已有 `graph` 环境，可直接 `conda activate graph`。

### 2. 配置 API

```bash
cp .env.example .env
```

编辑 `.env`，填入阿里云百炼 API Key：

```env
DASHSCOPE_API_KEY=sk-你的真实key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-flash
```

### 3. 启动 Web 工作台

```bash
python src/web/run.py
```

### 4. 浏览器访问

打开 `http://localhost:8089`，看到 4 阶段决策窗界面。

---

## n8n 工单通知（可选）

转人工场景下（用户说"投诉/经理/赔偿/转人工"等），系统会**自动**创建工单 + 通过 n8n 发邮件升级到二线客服/主管邮箱。**不配置 `N8N_WEBHOOK_URL` 则仅本地落库不发邮件，不影响主流程。**

### 链路概览

```text
用户投诉 → 意图识别（第 1+2 重保险）
       → ticket_transfer 命中
       → create_ticket()  ← 落 tickets.json
            └─ POST n8n webhook（同步，3s 超时，静默失败）
                 └─ n8n Set 节点格式化邮件正文
                      └─ n8n Send Email 节点 → 163 SMTP
                           → 客服邮箱（任意外部邮箱）
```

### 启用步骤

#### 1. 准备 n8n 实例

- 任意方式起 n8n（推荐 Docker：`docker run -p 5678:5678 docker.n8n.io/n8nio/n8n`）
- 浏览器访问 `http://localhost:5678` 完成 owner 账号注册

#### 2. 导入工作流

- n8n WebUI → Workflows → **Import from File**
- 选 [docs/n8n_workflow.json](docs/n8n_workflow.json)
- 工作流自动加载 3 节点：Webhook + Edit Fields + Send Email

#### 3. 配置 SMTP credential

打开 Send Email 节点：
- **fromEmail** 占位符 `<请填写发件邮箱>` → 替换为发件账号（例如 `your_sender@163.com`）
- **toEmail** 占位符 `<请填写客服收件邮箱>` → 替换为接收方
- **SMTP credential** 重新创建：

| 字段 | 163 邮箱示例 |
|:----|:----|
| Host | `smtp.163.com` |
| Port | `465` |
| SSL/TLS | ON |
| User | 发件邮箱完整地址 |
| Password | **163 SMTP 授权码**（不是登录密码，需在 163 邮箱「设置 → POP3/SMTP/IMAP」单独生成） |

#### 4. 激活工作流

n8n WebUI 工作流右上角的 **Active toggle** 切到 ON（n8n 2.x 版本可能显示为 **Publish**）。激活后 webhook 长期生效。

#### 5. 在 `.env` 配置 webhook URL

```env
N8N_WEBHOOK_URL=http://localhost:5678/webhook/ticket-created
```

> 本地部署时使用 `localhost`；如需远程访问，请替换为实际服务器地址。

#### 6. 验证集成

```bash
# 单独跑 ticket 工具冒烟（直接调 create_ticket → 验证 webhook + 邮件）
python src/tools/ticket.py

# Web 端端到端
python src/web/run.py
# 浏览器发投诉消息："我要投诉！要求经理出面赔偿！"
# → 收件邮箱应收到含完整对话历史 + 工单号 + 关联订单号的邮件
```

### Webhook payload 字段

| 字段 | 来源 | 示例 |
|:----|:----|:----|
| `ticket_id` | 项目代码生成 `WT+YYYYMMDDHHMMSS+3位随机` | `WT20260621171357836` |
| `user_id` | LangGraph 会话 thread_id | `user_001` |
| `order_id` | 正则 `JD\d{6,}` 从对话历史抓取 | `JD20240610003` 或 `""` |
| `user_query` | 用户本轮原始消息（不做摘要）| 完整投诉原话 |
| `conversation_history` | LangGraph state 提取 user+ai messages | 多行 `[HH:MM] 角色：内容` |
| `created_at` | `datetime.now().isoformat()` | `2026-06-21T17:13:57` |

详细架构与故障策略见 [docs/n8n_workflow.md](docs/n8n_workflow.md)。
