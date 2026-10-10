# Enterprise AI Agent 智能助手平台

> 企业级 AI Agent 平台：ReAct 推理 + RAG 知识库 + 对话记忆 + 多Agent团队协作 + 全链路治理（RBAC / 限流 / 配额 / 审计 / 脱敏）

## 核心能力

| 能力 | 说明 |
|:---|:---|
| **ReAct 推理循环** | LangGraph 编排 prepare → reason ↔ act → answer 五节点循环 |
| **RAG 知识库** | Milvus 向量检索 + pgvector 语义排序，支持 PDF/Markdown 入库 |
| **对话记忆** | PostgreSQL + pgvector 存储多轮对话与长期记忆 |
| **多Agent团队协作** | 主管拆任务 → 并行执行 → 共识检测 → 仲裁裁决 → 终审汇总 |
| **委派 Delegation** | 执行者中途转活给更适合的同伴，五道闸治理 + chain 防死循环 |
| **工具治理** | 三层包装：Guard(并发/超时) → Retry(指数退避) → Runner(执行) |
| **RBAC 权限** | MySQL 用户/角色/权限体系，知识库按授权过滤、字段脱敏 |
| **消息编排** | 中心化编排器：校验 → 限流 → 配额 → 防环 → 投递 → 观测 |
| **HITL 人机协同** | 工具审批 / 仲裁升级 / 终审修改，人工可介入任意环节 |
| **MCP 协议** | 标准化外部工具接入，JSON-RPC 调用 |

## 架构总览

```
入口层（main.py）         命令分发：chat / serve / ingest / stats / mcp-server / web-demo / offboard
   │
Agent 层（LangGraph）     ReAct 五节点循环 + Planning 规划执行
   │
能力层（Skills + Tools）   技能 SOP + 工具 Function Calling + MCP 外部工具
   │
知识层（RAG + Memory）    Milvus 知识库 + PostgreSQL 记忆库
   │
治理层（Orchestration）   消息总线 + 限流 + 配额 + 防环 + 观测链路
   │
企业层（Enterprise）      RBAC 权限 / 审计日志 / 数据脱敏 / 离职处置
```

## 快速开始

### 环境要求

- Python >= 3.10
- PostgreSQL + pgvector（对话记忆与业务表）
- Milvus 2.5+（知识库向量检索）
- MySQL 8.0+（企业身份与权限库，可选）

### 安装

```bash
# 克隆仓库
git clone https://github.com/xiexindong/ai_agent_project.git
cd ai_agent_project

# 安装依赖
pip install -r requirements.txt

# 复制配置文件并填写密钥
cp .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY / ZHIPU_API_KEY 等
```

### 启动数据库

```bash
# PostgreSQL + pgvector
docker run -d --name pg -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg16

# Milvus Standalone
docker run -d --name milvus -p 19530:19530 milvusdb/milvus:v2.5.18
```

### 运行

```bash
# 交互式对话
python main.py chat

# 启动 API 服务
python main.py serve

# 知识库入库
python main.py ingest docs/

# 查看知识库统计
python main.py stats

# MCP Server 独立启动
python main.py mcp-server

# 前端预览（Mock 数据，不依赖数据库）
python main.py web-demo

# 离职数据处置（跨库对账）
python main.py offboard --apply
```

## 项目结构

```
ai_agent_project/
├── main.py                    # 项目入口，argparse 子命令分发
├── config.py                  # 配置单例（.env 加载 + LangSmith 追踪）
├── modules/
│   ├── agent/                 # ReAct Agent（LangGraph 编排 + LLM 调用 + Planning）
│   ├── api/                   # FastAPI 服务端（SSE 流式 + 多Agent团队接口）
│   ├── common/                # 日志、异常、安全校验、token 计数
│   ├── enterprise/            # RBAC 权限、审计日志、脱敏、离职处置
│   ├── eval/                  # 评测集与性能基线
│   ├── harness/               # 工具治理（并发/超时/重试/沙箱）
│   ├── hitl/                  # 人机协同（审批/仲裁/终审）
│   ├── mcp/                   # MCP 协议适配
│   ├── memory/                # 对话记忆（PG + pgvector）
│   ├── multiagent/            # 多Agent团队（Harness/委派/编排/共识）
│   ├── multimodal/            # 多模态输入（图片）
│   ├── orchestration/         # 中心化编排器（消息总线/限流/配额/防环/Tracer）
│   ├── pipeline/              # 数据管道
│   ├── prompt/               # Prompt 模板与组装
│   ├── rag/                   # RAG 检索（Milvus + 重排序）
│   ├── skills/                # 技能 SOP（意图匹配 + 路由）
│   └── tools/                 # 工具注册与执行
├── tests/                     # pytest 测试（不依赖数据库/密钥/网络）
├── docs/                      # 架构文档
├── docker-compose.yml         # 一键启动全部依赖
├── Dockerfile
├── .env.example               # 配置模板
└── pyproject.toml             # 工程配置
```

## 测试

```bash
# 全部用例（不需要数据库/密钥/网络）
pytest

# 带详细输出
pytest tests/test_harness.py -v
```

## 配置说明

核心配置项（完整见 `.env.example`）：

| 配置 | 默认值 | 说明 |
|:---|:---|:---|
| `DEEPSEEK_API_KEY` | — | LLM 对话模型密钥 |
| `ZHIPU_API_KEY` | — | Embedding 向量模型密钥 |
| `RBAC_ENABLED` | true | 是否走 MySQL 权限体系 |
| `MULTIAGENT_ENABLED` | true | 多Agent协作总开关 |
| `MULTIAGENT_DELEGATION_ENABLED` | false | 委派总开关 |
| `MULTIAGENT_MAX_WORKERS` | 4 | 并行 Worker 数 |
| `MULTIAGENT_TIMEOUT_SECONDS` | 120 | 团队运行总超时 |

## 技术栈

| 层 | 技术 |
|:---|:---|
| LLM | DeepSeek（兼容 OpenAI 接口） |
| Embedding | 智谱 embedding-2（1024 维） |
| Agent 编排 | LangGraph |
| 向量检索 | Milvus 2.5+ |
| 对话记忆 | PostgreSQL + pgvector |
| 权限体系 | MySQL（RBAC 五表模型） |
| API 服务 | FastAPI + uvicorn + SSE |
| 工具协议 | MCP（Model Context Protocol） |
| 测试 | pytest |

## 文档

详细架构文档见 [docs/](./docs/README.md)，按"先跑通 → 再深入 → 再扩展"顺序组织。

## License

MIT
