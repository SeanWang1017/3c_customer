# 3C 电商智能客服 Agent 系统

> 技术栈：LLaMA Factory + LangChain + FAISS + FastAPI + QLoRA
> 云端模型：qwen-flash

---

## 这个项目做什么

用户通过自然语言和智能客服对话，系统辅助客服生成回答，**经过客服人工审核后**再发送给客户：

- **商品咨询 / 推荐** — "推荐一款 3000 以内的游戏手机" → 检索商品库 → **qwen-flash** 生成建议回答 → 客服审核后发出
- **售后政策问答** — "耳机拆封还能退吗" → 检索 FAQ → **qwen-flash** 生成建议回答 → 客服审核后发出
- **订单 / 物流查询** — "我的快递到哪了" → 调用查询工具 → 客服确认后发出
- **转人工** — "转人工" → 路由识别 → 直接转人工

一句话：**本地模型做意图分类 + qwen-flash 生成回复 + 人工审核后发出。**

---

## 技术选型

| 模块 | 选什么 | 用途 |
|:----|:-------|:-----|
| 基座模型 | Qwen2.5-0.5B | 本地微调（意图分类） |
| 微调框架 | LLaMA Factory | QLoRA 微调 |
| 微调方法 | QLoRA int4 | 适配 8GB 显存 |
| **云端 LLM** | **qwen-flash（阿里云百炼）** | **最终回复生成** |
| Agent 框架 | LangChain 1.x（`create_agent` + `ToolCallLimitMiddleware`） | 意图路由 + 工具调用 |
| 向量库 | FAISS（IndexFlatIP） | RAG 检索 |
| Embedding | BGE-small-zh-v1.5 | 中文语义检索 |
| 前端 | FastAPI + SSE 流式 | 客服工作台 |
| 部署 | 本地运行 | 本地 conda 环境 |

**LLM 角色分工：**

- **本地 Qwen2.5-0.5B + QLoRA** → 意图分类（5 分类短标签，89.8% 准确率）
- **云端 qwen-flash** → 最终客服回复生成（结合 RAG + 工具结果）

---

## 数据从哪来

### 训练数据

数据来源：**LLM 生成 + JDDC 真实客服对话** 融合

| 来源 | 条数 | 说明 |
|:----|:----:|:------|
| LLM 生成（5 场景） | 2500 | 商品咨询/推荐/订单查询/售后咨询/售后处理 |
| JDDC 真实客服对话 | 982 | [京东对话挑战赛（JDDC）](https://github.com/EndlessLethe/jddc2019-3th-retrieve-model#) 真实客服数据 |
| **融合总量** | **3482** | 源数据（sharegpt 格式）；提取意图分类样本后得 train 2028 / val 227 / test 245（alpaca 格式） |

格式是 ShareGPT（多轮对话数组）。JDDC 数据经过脱敏处理，保留真实客服对话风格。

### 知识库数据

| 数据 | 数量 | 来源 | 用途 |
|:----|:----:|:-----|:-----|
| 商品数据 | 3929 条 | 京东 3C 商品清洗 | RAG 检索用（products.index） |
| FAQ 数据 | **33 条** | 人工编写（含三包政策 FQ031-033） | RAG 检索用（faq.index） |

### 运行时数据

| 数据 | 数量 | 格式 | 用途 |
|:----|:----:|:-----|:-----|
| 订单数据 | 10 条 | JSON | query_order 工具 |
| 工单数据 | 7 条 | JSON | 工单工具 |

---

## 微调参数

### QLoRA int4（Qwen2.5-0.5B）

- 模型加载：~1GB（int4 量化，540M 参数）
- 训练过程：~3-4GB
- 剩余显存：~4-5GB

### 关键参数

```
LoRA: rank=16, alpha=32, dropout=0.05, target=all layers
训练: batch=2, grad_accum=4, lr=3e-5, epoch=3, cutoff=512
调度: cosine scheduler, warmup=50 steps
训练提示: train_on_prompt=false, enable_thinking=false
保存: 每 200 steps 一次
```

### 完整训练配置

```yaml
model_name: models/Qwen2.5-0.5B
finetuning_type: lora
quantization_bit: 4
quantization_method: bnb
template: qwen

batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 3e-5
num_train_epochs: 3
cutoff_len: 512

lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: all

warmup_steps: 50
lr_scheduler_type: cosine
save_steps: 200
compute_type: fp16

train_on_prompt: false
enable_thinking: false

dataset:
  - customer_service_intent_train
  - customer_service_intent_val
  - customer_service_intent_test
dataset_dir: data
```

### 评测指标

- 意图识别准确率 **89.8%（220/245）**，超过 85% 验收线
- RAG 检索通过率 **93.3%（14/15）**
- 端到端综合评测 **91.0 分（455/500）**
- 详见 [evaluation.md](evaluation.md)

---

## 云端 LLM 配置

- **模型**: `qwen-flash`（阿里云百炼 OpenAI 兼容 API）
- **base_url**: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- **API Key**: 配置在 `.env` 的 `DASHSCOPE_API_KEY`
- **temperature**: 0.6
- **max_tokens**: 1500

### 配置示例（.env 文件）

```env
DASHSCOPE_API_KEY=sk-xxx
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-flash
```

---

## Agent 工作流

### 4 重保险意图路由

```
用户输入
  ↓
[第 1 重保险] 本地 Qwen2.5-0.5B + LoRA → 5 分类（89.8%）
  ↓
[第 2 重保险] 17 条关键词规则 → 修正 product_intro ⇄ policy_qa 互混 / 三包等
  ↓
[第 3 重保险] 置信度<0.7 + 第 2 重没改判 → 调云端 qwen-flash 重新分类（可选，llm 参数 None 时跳过）
  ↓
[路由分发]
  ├─ order_query        → query_order 工具 → orders.json
  ├─ product_intro      → search_product 工具 → FAISS 商品库
  ├─ product_recommend  → search_product 工具 → FAISS 商品库
  ├─ policy_qa          → search_faq 工具 → FAISS FAQ 库
  └─ ticket_transfer    → 直接转人工固定回复（不调工具）
  ↓
[云端 qwen-flash] 看 hint + 工具结果 → 生成自然语言回复
  ↓
[第 4 重保险] 客服人工审核（FastAPI 前端 UI）→ 确认 / 修改 / 驳回
  ↓
发送给客户
```

### 已实现的 3 个工具

| 工具 | 触发场景 | 数据源 |
|:----|:---------|:-------|
| query_order | "查订单 JD20240610001" | data/orders/orders.json |
| search_faq | "三包政策 / 激活后能退吗" | FAISS faq.index（33 条） |
| search_product | "推荐游戏本 / iPhone 怎么样" | FAISS products.index（3929 条） |

### 工程加固

- **ToolCallLimitMiddleware(run_limit=5, exit_behavior=continue)** — 防 LLM 死循环，超量后用已有 tool 数据收尾
- **RAG 子进程隔离** — `scripts/rag_query.py` 独立进程跑 FAISS+sentence_transformers，绕开 langchain 主进程 segfault
- **InMemorySaver** — 多轮对话支持，按 thread_id 记忆上下文

### RAG 流程

```
用户问题 → BGE-small-zh-v1.5 编码（512 维向量）
                    ↓
            FAISS IndexFlatIP 检索（Top-3，cosine 相似度）
                    ↓
            返回相似文档 → 注入 LLM prompt → 生成回答
```

---

## 端到端测试用例

跑：

```bash
python src/agent/customer_service_agent.py
```

测试用例：

| Query | 路由 | 工具 | 结果 |
|:----|:----|:----|:----:|
| 我的订单 JD20240610001 到哪了？ | order_query | query_order | ✅ 已签收数据 |
| 激活后能退吗？ | policy_qa | search_faq | ✅ 引用 FAQ |
| 推荐一款游戏本 | product_recommend | search_product | ✅ 3 款推荐 |
| 我要投诉转人工 | ticket_transfer | None | ✅ 转人工固定回复 |
| 三包政策是什么？ | policy_qa | search_faq | ✅ 引用 FQ031+FQ032 |

---

## FastAPI Web 界面

启动：

```bash
python src/web/run.py
```

浏览器开 `http://localhost:8089`。

**界面布局**：

- **左栏**：客户对话（消息气泡 + 输入框 + 清空按钮）
- **右栏**：智能体决策窗（4 阶段实时更新）

**第 4 重保险 客服审核 UI**：每条 AI 回复下自动出现"✅ 确认发送 / ❌ 驳回"按钮。

---

## 项目结构

```
3C_customer_agent/
├── config/train_lora.yaml         # 训练配置
├── data/
│   ├── processed/                 # 商品数据（jd_all_cleaned.xlsx，3929 条）
│   ├── intent/                    # 训练数据（JSONL）
│   ├── faq/                       # FAQ（faq.json，33 条）
│   ├── orders/                    # 订单/工单（orders.json + tickets.json）
│   └── vector_store/              # FAISS 索引（faq.index + products.index）
├── saves/qwen2.5-0.5b-intent-lora/   # LoRA 输出
├── models/
│   ├── Qwen2.5-0.5B/              # 基座模型（ModelScope）
│   └── embeddings/bge-small-zh-v1.5/ # 中文 Embedding
├── scripts/
│   ├── generate_intent_data.py
│   └── rag_query.py               # RAG 子进程入口
├── test/
│   ├── test_intent.py             # 意图分类评测
│   ├── test_rag.py                # RAG 端到端评测
│   └── eval_e2e.py                # 端到端综合评测
├── src/
│   ├── intent/
│   │   ├── classifier.py          # 第 1 重保险：本地模型加载 + 分类
│   │   └── postprocess.py         # 第 2 重保险：关键词规则（17 条）
│   ├── rag/                       # 向量检索（embedder/retriever/build_index）
│   ├── tools/
│   │   ├── order_query.py         # 订单纯函数
│   │   └── langchain_tools.py     # 3 个 @tool 包装
│   ├── agent/
│   │   ├── customer_service_agent.py  # 主 Agent
│   │   ├── intent_router.py           # 4 重保险意图路由
│   │   └── system_prompt.txt          # 系统提示词
│   └── web/                       # FastAPI 后端
│       ├── app.py                     # SSE 4 阶段流式
│       ├── static/index.html          # 单文件前端
│       └── run.py                     # 启动脚本
└── docs/                          # 文档（含本 mindmap）
```

---

## 环境

```
conda 环境: graph（开发环境名，可任意命名）
GPU: 8GB+ 显存（仅训练需要；推理/启动 Web 不需要 GPU）
Python: 3.10+
```

```bash
conda activate graph
pip install -r requirements.txt
cp .env.example .env  # 填入 DASHSCOPE_API_KEY

# 跑 Web 工作台（推荐）
python src/web/run.py
# → 浏览器开 http://localhost:8089

# 跑 Agent 命令行 Demo
python src/agent/customer_service_agent.py
```

---

## 关键技术决策

| 决策 | 原因 |
|:----|:----|
| 阿里云百炼 base_url 用 `/compatible-mode/v1` | `/api/v1` 走百炼原生 API，OpenAI 兼容模式会 404 |
| 云端模型选 `qwen-flash` | 在便宜与稳定之间取得平衡 |
| RAG 工具走子进程 | langchain 主进程同时 import faiss+sentence_transformers 会 segfault，importlib 懒加载不够 |
| 意图模型只跑 1 次 | 多意图 query 由 LLM 自主补调（hint 路由 + 工具列表统一暴露） |
| temperature=0.6, max_tokens=1500 | 客服场景平衡多样性与准确性，避免截断 |
| 4 重保险而非单层 | 第 1 重失败 第 2 重救场 → 第 2 重救不了 第 3 重救场 → 第 3 重也救不了 第 4 重人工救场 |
