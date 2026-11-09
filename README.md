# 苍穹外卖 Agent 3.0

该服务是基于 FastAPI、LangGraph 和本地 DeepSeek-R1 的业务 Agent，而不是简单的模型接口转发。

## 能力

- LangGraph：多 Agent 路由、状态管理、工具执行、失败重试和确认流程
- 多 Agent 路由：点餐 Agent 与经营 Agent
- 工作流：准备、路由、规划、工具执行、反思重试、结果生成
- Java Tool Gateway：实时读取菜品、用户订单画像、经营指标和备餐预测
- RAG：本地知识入库、向量检索和业务规则增强
- 记忆：Redis 短期会话，SQLite 降级与用户长期偏好
- Human-in-the-loop：营销、菜单和购物车方案先生成任务，再人工确认
- 异步任务：创建、查询、执行、失败状态和结果持久化
- 安全与可观测性：工具白名单、内部令牌、提示词注入拦截、审计、反馈、Prometheus 指标

## 本地启动

```powershell
cd "D:\Code\Resume items\CangQiongWaiMai\ai-agent-service"
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8001
```

依赖服务：Java 后端 `8080`、Ollama `11434`，Redis 可选。接口文档为 `http://127.0.0.1:8001/docs`。

## 自动评测

```powershell
$env:AGENT_EVAL_URL="http://127.0.0.1:8001"
.\.venv\Scripts\python.exe eval_runner.py
```

## Docker 与监控

在项目根目录执行 `docker compose -f docker-compose.agent.yml up --build`。Prometheus 地址为 `http://127.0.0.1:9090`。
