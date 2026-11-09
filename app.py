import json
import hashlib
import math
import os
import re
import sqlite3
import time
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional, TypedDict

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Response
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

try:
    import redis
except ImportError:  # pragma: no cover
    redis = None

load_dotenv()
app = FastAPI(title="Sky Take Out Agent Platform", version="2.0.0")
metrics = Counter()
DB_PATH = os.getenv("AGENT_DB_PATH", os.path.join(os.path.dirname(__file__), "agent.db"))


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("create table if not exists sessions (session_id text primary key, history text not null, updated_at integer not null)")
        connection.execute("create table if not exists feedback (id integer primary key autoincrement, actor_id integer, session_id text, rating integer, comment text, created_at integer)")
        connection.execute("create table if not exists agent_runs (id integer primary key autoincrement, session_id text, role text, intent text, tool_calls text, latency_ms integer, created_at integer)")
        connection.execute("create table if not exists user_profiles (actor_id integer primary key, preferences text not null, updated_at integer not null)")
        connection.execute("create table if not exists knowledge_documents (id integer primary key autoincrement, title text not null, content text not null, embedding text not null)")
        connection.execute("create table if not exists agent_tasks (task_id text primary key, actor_id integer, role text, task_type text, payload text, status text, result text, created_at integer, updated_at integer)")


init_db()


def text_embedding(text: str, size: int = 128) -> List[float]:
    vector = [0.0] * size
    normalized = "".join(text.lower().split())
    for index in range(max(1, len(normalized) - 1)):
        digest = hashlib.sha256(normalized[index:index + 2].encode("utf-8")).digest()
        vector[int.from_bytes(digest[:4], "big") % size] += 1.0
    length = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / length for value in vector]


def seed_knowledge() -> None:
    documents = [
        ("点餐搭配原则", "两人点餐建议一荤一素一汤，预算不足时优先保证主食和蛋白质，忌辣用户过滤麻辣和香辣菜品。"),
        ("订单完成率处理", "订单样本少于10笔时只提示观察，不判定正式异常；完成率低于80%且样本充足时需要分析取消原因。"),
        ("菜单优化原则", "菜单优化应结合销量、毛利和复购率。Agent只能生成提案，菜品上下架和套餐创建必须由管理员确认。"),
    ]
    with sqlite3.connect(DB_PATH) as connection:
        for doc_id, (title, content) in enumerate(documents, 1):
            connection.execute("insert into knowledge_documents(id, title, content, embedding) values(?, ?, ?, ?) on conflict(id) do update set title=excluded.title, content=excluded.content, embedding=excluded.embedding", (doc_id, title, content, json.dumps(text_embedding(content))))


seed_knowledge()


class AgentRequest(BaseModel):
    role: str = Field(pattern="^(user|admin)$")
    actor_id: int
    session_id: Optional[str] = None
    message: str = Field(min_length=1, max_length=2000)
    intent: str = "chat"
    context: Dict[str, Any] = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    actor_id: int
    session_id: str
    rating: int = Field(ge=-1, le=1)
    comment: str = Field(default="", max_length=500)


class TaskRequest(BaseModel):
    actor_id: int
    role: str = Field(pattern="^(user|admin)$")
    task_type: str
    action: Dict[str, Any] = Field(default_factory=lambda: {"type": "REPORT_ONLY"})


class KnowledgeRequest(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    content: str = Field(min_length=1, max_length=5000)


class AgentState(TypedDict, total=False):
    request: AgentRequest
    session_id: str
    history: List[Dict[str, Any]]
    plan: List[str]
    tool_args: Dict[str, Dict[str, Any]]
    tool_results: Dict[str, Any]
    model_result: Dict[str, Any]
    result: Dict[str, Any]
    started_at: float
    specialist: str
    critique: Dict[str, Any]
    attempts: int


def tool_schema(name: str, description: str, properties: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties},
        },
    }


TOOL_SCHEMAS = [
    tool_schema("search_available_dishes", "Search enabled dishes by budget, taste and keywords", {"budget": {"type": "number"}, "avoid_spicy": {"type": "boolean"}, "keywords": {"type": "array", "items": {"type": "string"}}}),
    tool_schema("build_budget_plan", "Build a complete menu within a total budget", {"budget": {"type": "number"}, "people": {"type": "integer"}}),
    tool_schema("analyze_business_metrics", "Analyze turnover, orders and top selling dishes", {}),
    tool_schema("detect_business_anomalies", "Detect low completion rate, falling turnover and concentration risk", {}),
    tool_schema("generate_marketing_plan", "Generate a marketing proposal from current business metrics", {"goal": {"type": "string"}}),
    tool_schema("optimize_menu", "Find menu optimization opportunities from sales data", {}),
    tool_schema("get_user_profile", "Load preferences inferred from order history and feedback", {}),
    tool_schema("retrieve_knowledge", "Retrieve operational knowledge relevant to the request", {"query": {"type": "string"}}),
    tool_schema("forecast_prep_demand", "Forecast preparation quantities from recent sales", {}),
]


def redis_client():
    if redis is None or not os.getenv("REDIS_URL"):
        return None
    try:
        client = redis.from_url(os.environ["REDIS_URL"], decode_responses=True, socket_timeout=1)
        client.ping()
        return client
    except Exception:
        return None


def load_history(session_id: str) -> List[Dict[str, Any]]:
    client = redis_client()
    if client:
        raw = client.get("agent:session:" + session_id)
        return json.loads(raw) if raw else []
    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute("select history from sessions where session_id = ?", (session_id,)).fetchone()
    return json.loads(row[0]) if row else []


def save_history(session_id: str, history: List[Dict[str, Any]]) -> None:
    history = history[-30:]
    client = redis_client()
    if client:
        client.setex("agent:session:" + session_id, 86400, json.dumps(history, ensure_ascii=False))
    else:
        with sqlite3.connect(DB_PATH) as connection:
            connection.execute("insert into sessions(session_id, history, updated_at) values(?, ?, ?) on conflict(session_id) do update set history=excluded.history, updated_at=excluded.updated_at", (session_id, json.dumps(history, ensure_ascii=False), int(time.time())))


def number_after(words: List[str], text: str, default: float) -> float:
    import re
    for word in words:
        match = re.search(word + r"[^0-9]{0,4}(\d+(?:\.\d+)?)", text)
        if match:
            return float(match.group(1))
    return default


def java_tool_get(path: str) -> Any:
    base_url = os.getenv("JAVA_TOOL_BASE_URL", "http://127.0.0.1:8080/internal/agent").rstrip("/")
    response = httpx.get(base_url + path, headers={"X-Agent-Token": os.getenv("JAVA_TOOL_TOKEN", "sky-agent-local-token")}, timeout=10)
    response.raise_for_status()
    envelope = response.json()
    if envelope.get("code") != 1:
        raise RuntimeError(envelope.get("msg") or "Java tool failed")
    return envelope.get("data")


def retrieve_knowledge(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    target = text_embedding(query)
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute("select id, title, content, embedding from knowledge_documents").fetchall()
    scored = []
    for doc_id, title, content, raw_embedding in rows:
        score = sum(left * right for left, right in zip(target, json.loads(raw_embedding)))
        scored.append({"id": doc_id, "title": title, "content": content, "score": round(score, 4)})
    return sorted(scored, key=lambda item: item["score"], reverse=True)[:limit]


def load_user_preferences(actor_id: int) -> Dict[str, Any]:
    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute("select preferences from user_profiles where actor_id=?", (actor_id,)).fetchone()
    return json.loads(row[0]) if row else {}


def update_user_preferences(request: AgentRequest) -> None:
    if request.role != "user":
        return
    preferences = load_user_preferences(request.actor_id)
    budget = number_after(["预算", "不超过", "以内", "budget"], request.message.lower(), 0)
    if budget:
        preferences["recentBudget"] = budget
    if any(word in request.message.lower() for word in ["不辣", "不吃辣", "忌辣", "不能吃辣", "no spicy"]):
        preferences["avoidSpicy"] = True
    preferences["lastRequest"] = request.message
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("insert into user_profiles(actor_id, preferences, updated_at) values(?, ?, ?) on conflict(actor_id) do update set preferences=excluded.preferences, updated_at=excluded.updated_at", (request.actor_id, json.dumps(preferences, ensure_ascii=False), int(time.time())))


def execute_tool(name: str, args: Dict[str, Any], request: AgentRequest) -> Any:
    context = request.context
    dishes = context.get("dishes") or []
    if name in ("search_available_dishes", "build_budget_plan") and not dishes:
        dishes = java_tool_get("/tools/dishes")
    if name == "search_available_dishes":
        budget = float(args.get("budget") or number_after(["预算", "不超过", "以内", "budget"], request.message.lower(), 999999))
        avoid_spicy = bool(args.get("avoid_spicy")) or any(x in request.message.lower() for x in ["不辣", "不吃辣", "忌辣", "不能吃辣", "no spicy"])
        keywords = args.get("keywords") or []
        selected = []
        for dish in dishes:
            price = float(dish.get("price") or 0)
            detail = json.dumps(dish, ensure_ascii=False)
            if price > budget or (avoid_spicy and any(x in detail for x in ["辣", "麻", "椒", "水煮"])):
                continue
            if keywords and not any(str(x) in detail for x in keywords):
                continue
            selected.append(dish)
        return selected[:8]
    if name == "build_budget_plan":
        budget = float(args.get("budget") or number_after(["预算", "不超过", "以内", "budget"], request.message.lower(), 100))
        people = int(args.get("people") or number_after(["人数", "个人", "人", "people"], request.message.lower(), 2))
        avoid_spicy = any(x in request.message.lower() for x in ["不辣", "不吃辣", "忌辣", "不能吃辣", "no spicy"])
        spicy_terms = ["辣", "麻", "椒", "水煮"]
        candidates = [dish for dish in dishes if not avoid_spicy or not any(term in json.dumps(dish, ensure_ascii=False) for term in spicy_terms)]
        category_order = [18, 19, 21, 12, 16, 17, 20, 11]
        candidates.sort(key=lambda x: (category_order.index(x.get("categoryId")) if x.get("categoryId") in category_order else 99, -float(x.get("price") or 0)))
        plan, total = [], 0.0
        for category in category_order:
            for dish in candidates:
                price = float(dish.get("price") or 0)
                if dish.get("categoryId") == category and price > 0 and total + price <= budget:
                    plan.append(dish)
                    total += price
                    break
            if len(plan) >= max(3, people + 1):
                break
        return {"people": people, "budget": budget, "total": total, "remaining": round(budget - total, 2), "dishes": plan}
    if name == "analyze_business_metrics":
        if not context.get("orders"):
            return java_tool_get("/tools/business-metrics")
        return {k: context.get(k) for k in ["dateRange", "turnover", "orders", "salesTop10"]}
    if name == "detect_business_anomalies":
        orders = context.get("orders") or {}
        rate = float(orders.get("orderCompletionRate") or 0)
        count_values = [int(float(x)) for x in str(orders.get("orderCountList") or "").split(",") if x.strip()]
        total_orders = sum(count_values)
        anomalies = []
        if total_orders < 10:
            anomalies.append({"level": "info", "type": "LOW_SAMPLE", "message": "订单样本少于10笔，当前仅建议持续观察"})
        elif rate < 0.8:
            anomalies.append({"level": "high", "type": "LOW_COMPLETION_RATE", "message": "订单完成率低于80%"})
        turnover = (context.get("turnover") or {}).get("turnoverList") or ""
        values = [float(x) for x in str(turnover).split(",") if x.strip()]
        if len(values) >= 2 and values[-1] < values[-2] * 0.8:
            anomalies.append({"level": "medium", "type": "TURNOVER_DROP", "message": "最近营业额环比下降超过20%"})
        return anomalies or [{"level": "info", "type": "NORMAL", "message": "未发现显著经营异常"}]
    if name == "generate_marketing_plan":
        return {"goal": args.get("goal") or request.message, "proposal": ["将热销菜与高毛利菜组成套餐", "对低峰时段发放满减券", "对30天未下单用户定向召回"], "requiresConfirmation": True}
    if name == "optimize_menu":
        top = context.get("salesTop10") or {}
        return {"topSelling": top, "proposal": ["保留热销菜的主展示位", "将关联菜品组合为套餐", "对无销量菜品执行降价测试或下架评审"], "requiresConfirmation": True}
    if name == "get_user_profile":
        profile = java_tool_get("/tools/user-profile/" + str(request.actor_id))
        profile["learnedPreferences"] = load_user_preferences(request.actor_id)
        return profile
    if name == "retrieve_knowledge":
        return retrieve_knowledge(str(args.get("query") or request.message))
    if name == "forecast_prep_demand":
        return java_tool_get("/tools/prep-forecast")
    raise ValueError("unknown tool: " + name)


def fallback_plan(request: AgentRequest) -> List[str]:
    intent = request.intent.lower()
    message = request.message
    if request.role == "admin":
        if "异常" in message or "anomaly" in intent:
            return ["analyze_business_metrics", "detect_business_anomalies"]
        if "营销" in message or "marketing" in intent:
            return ["analyze_business_metrics", "generate_marketing_plan"]
        if "菜单" in message or "menu" in intent:
            return ["analyze_business_metrics", "optimize_menu"]
        return ["analyze_business_metrics"]
    if "方案" in message or "搭配" in message or "plan" in intent:
        return ["build_budget_plan"]
    return ["search_available_dishes"]


async def model_plan(request: AgentRequest, history: List[Dict[str, Any]]) -> tuple[List[str], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        return fallback_plan(request), {}, {}
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    if "11434" in base_url or model.startswith("deepseek-r1"):
        if request.role == "user":
            allowed = ["build_budget_plan"] if "plan" in request.intent else ["search_available_dishes", "build_budget_plan"]
        else:
            intent_tools = {
                "business_insights": ["analyze_business_metrics"],
                "anomaly_detect": ["analyze_business_metrics", "detect_business_anomalies"],
                "menu_optimize": ["analyze_business_metrics", "optimize_menu"],
                "marketing_plan": ["analyze_business_metrics", "generate_marketing_plan"],
            }
            allowed = intent_tools.get(request.intent, ["analyze_business_metrics", "detect_business_anomalies", "generate_marketing_plan", "optimize_menu"])
        prompt = (
            "You are a tool planner. Return one JSON object only, without markdown: "
            '{"tools":[{"name":"tool_name","arguments":{}}]}. '
            "Allowed tools: " + ", ".join(allowed) + ". "
            "Role=" + request.role + ", intent=" + request.intent + ", request=" + request.message
        )
        payload = {
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": request.message}],
        }
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(base_url + "/chat/completions", headers={"Authorization": "Bearer " + api_key}, json=payload)
                response.raise_for_status()
            message = response.json()["choices"][0]["message"]
            content = re.sub(r"<think>.*?</think>", "", message.get("content") or "", flags=re.S).strip()
            parsed = json.loads(content[content.find("{"):content.rfind("}") + 1])
            calls = [item for item in parsed.get("tools", []) if item.get("name") in allowed]
            names = [item["name"] for item in calls]
            if request.intent in ("anomaly_detect", "menu_optimize", "marketing_plan") and "analyze_business_metrics" not in names:
                names.insert(0, "analyze_business_metrics")
            arguments = {
                item["name"]: item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
                for item in calls
            }
            return names or fallback_plan(request), arguments, message
        except Exception:
            metrics["model_failures"] += 1
            return fallback_plan(request), {}, {}
    payload = {
        "model": model, "temperature": 0.1,
        "messages": [{"role": "system", "content": "You are an agent planner. Select business tools. Never execute mutations."}] + history[-8:] + [{"role": "user", "content": request.message}],
        "tools": TOOL_SCHEMAS, "tool_choice": "auto"
    }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(base_url + "/chat/completions", headers={"Authorization": "Bearer " + api_key}, json=payload)
            response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        names = [call["function"]["name"] for call in calls]
        arguments = {call["function"]["name"]: json.loads(call["function"].get("arguments") or "{}") for call in calls}
        return names or fallback_plan(request), arguments, message
    except Exception:
        metrics["model_failures"] += 1
        return fallback_plan(request), {}, {}


async def prepare_node(state: AgentState) -> AgentState:
    request = state["request"]
    blocked_patterns = ["ignore previous", "忽略之前", "system prompt", "系统提示词", "reveal secret", "泄露密钥"]
    if any(pattern in request.message.lower() for pattern in blocked_patterns):
        metrics["prompt_injection_blocked"] += 1
        raise HTTPException(status_code=400, detail="potential prompt injection detected")
    session_id = request.session_id or str(uuid.uuid4())
    return {**state, "session_id": session_id, "history": load_history(session_id), "started_at": time.time(), "attempts": 0}


async def router_node(state: AgentState) -> AgentState:
    specialist = "ordering_agent" if state["request"].role == "user" else "business_agent"
    return {**state, "specialist": specialist}


async def plan_node(state: AgentState) -> AgentState:
    plan, tool_args, model_result = await model_plan(state["request"], state["history"])
    request = state["request"]
    if request.role == "user":
        plan = ["get_user_profile", "retrieve_knowledge"] + plan
        tool_args.setdefault("retrieve_knowledge", {"query": request.message})
    else:
        plan = ["retrieve_knowledge"] + plan
        tool_args.setdefault("retrieve_knowledge", {"query": request.message})
        required_by_intent = {
            "anomaly_detect": ["analyze_business_metrics", "detect_business_anomalies"],
            "menu_optimize": ["analyze_business_metrics", "optimize_menu"],
            "marketing_plan": ["analyze_business_metrics", "generate_marketing_plan"],
            "prep_forecast": ["analyze_business_metrics", "forecast_prep_demand"],
        }
        plan.extend(required_by_intent.get(request.intent, []))
        if "forecast" in request.intent or "预测" in request.message:
            plan.append("forecast_prep_demand")
    plan = list(dict.fromkeys(plan))
    return {**state, "plan": plan, "tool_args": tool_args, "model_result": model_result}


async def tools_node(state: AgentState) -> AgentState:
    request, results = state["request"], {}
    for name in state["plan"]:
        try:
            results[name] = execute_tool(name, state.get("tool_args", {}).get(name, {}), request)
            metrics["tool_success"] += 1
        except Exception as exc:
            results[name] = {"error": str(exc)}
            metrics["tool_failure"] += 1
    return {**state, "tool_results": results, "attempts": state.get("attempts", 0) + 1}


async def critic_node(state: AgentState) -> AgentState:
    failures = [name for name, value in state["tool_results"].items() if isinstance(value, dict) and value.get("error")]
    decision = "retry" if failures and state.get("attempts", 0) < 2 else "pass"
    return {**state, "critique": {"decision": decision, "failedTools": failures}}


def critic_route(state: AgentState) -> str:
    return "planner" if state["critique"]["decision"] == "retry" else "response_builder"


def create_task(actor_id: int, role: str, task_type: str, payload: Dict[str, Any], status: str = "WAITING_CONFIRMATION") -> str:
    task_id = str(uuid.uuid4())
    now = int(time.time())
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("insert into agent_tasks(task_id, actor_id, role, task_type, payload, status, result, created_at, updated_at) values(?, ?, ?, ?, ?, ?, ?, ?, ?)", (task_id, actor_id, role, task_type, json.dumps(payload, ensure_ascii=False), status, "{}", now, now))
    return task_id


async def respond_node(state: AgentState) -> AgentState:
    request, tools = state["request"], state["tool_results"]
    suggestions: List[Dict[str, Any]] = []
    if "search_available_dishes" in tools:
        suggestions = tools["search_available_dishes"][:3]
    if "build_budget_plan" in tools:
        suggestions = tools["build_budget_plan"].get("dishes", [])
    if request.role == "user":
        names = "、".join(str(x.get("name")) for x in suggestions) or "暂无合适菜品"
        reply = "已完成菜品检索和预算校验，推荐：" + names
        action = {"type": "CONFIRM_CART", "requiresConfirmation": bool(suggestions)}
    else:
        reply = "经营分析已完成。" + json.dumps(tools, ensure_ascii=False)
        requires_confirmation = "generate_marketing_plan" in tools or "optimize_menu" in tools
        action = {"type": "CONFIRM_ADMIN_PROPOSAL" if requires_confirmation else "NONE", "requiresConfirmation": requires_confirmation}
    if action["requiresConfirmation"]:
        task_payload = {"type": "REPORT_ONLY", "proposal": tools}
        action["taskId"] = create_task(request.actor_id, request.role, action["type"], task_payload)
    result = {"reply": reply, "suggestions": [{"dishId": x.get("id"), "name": x.get("name"), "price": x.get("price"), "reason": "Agent工具筛选结果"} for x in suggestions], "action": action, "toolCalls": [{"name": x, "status": "success" if not isinstance(tools[x], dict) or "error" not in tools[x] else "failed"} for x in state["plan"]], "toolResults": tools, "sessionId": state["session_id"]}
    history = state["history"] + [{"role": "user", "content": request.message}, {"role": "assistant", "content": reply}]
    save_history(state["session_id"], history)
    update_user_preferences(request)
    latency_ms = int((time.time() - state["started_at"]) * 1000)
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("insert into agent_runs(session_id, role, intent, tool_calls, latency_ms, created_at) values(?, ?, ?, ?, ?, ?)", (state["session_id"], request.role, request.intent, json.dumps(state["plan"]), latency_ms, int(time.time())))
    metrics["requests"] += 1
    metrics["latency_ms_total"] += latency_ms
    return {**state, "result": result}


builder = StateGraph(AgentState)
builder.add_node("prepare", prepare_node)
builder.add_node("router", router_node)
builder.add_node("planner", plan_node)
builder.add_node("tool_executor", tools_node)
builder.add_node("critic", critic_node)
builder.add_node("response_builder", respond_node)
builder.add_edge(START, "prepare")
builder.add_edge("prepare", "router")
builder.add_edge("router", "planner")
builder.add_edge("planner", "tool_executor")
builder.add_edge("tool_executor", "critic")
builder.add_conditional_edges("critic", critic_route, {"planner": "planner", "response_builder": "response_builder"})
builder.add_edge("response_builder", END)
agent_graph = builder.compile()


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "version": "3.0.0", "graph": ["prepare", "router", "planner", "tool_executor", "critic", "response_builder"], "memory": "redis" if redis_client() else "sqlite-fallback", "model": os.getenv("LLM_MODEL") if os.getenv("LLM_API_KEY") else "rules-fallback"}


@app.post("/v1/agent/chat")
async def chat(request: AgentRequest) -> Dict[str, Any]:
    state = await agent_graph.ainvoke({"request": request})
    return state["result"]


@app.get("/v1/agent/sessions/{session_id}")
async def session(session_id: str) -> Dict[str, Any]:
    history = load_history(session_id)
    if not history:
        raise HTTPException(status_code=404, detail="session not found")
    return {"sessionId": session_id, "history": history}


@app.post("/v1/agent/feedback")
async def feedback(request: FeedbackRequest) -> Dict[str, Any]:
    item = {**request.model_dump(), "createdAt": int(time.time())}
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("insert into feedback(actor_id, session_id, rating, comment, created_at) values(?, ?, ?, ?, ?)", (request.actor_id, request.session_id, request.rating, request.comment, item["createdAt"]))
    metrics["positive_feedback" if request.rating > 0 else "negative_feedback"] += 1
    return {"accepted": True}


@app.get("/v1/agent/profiles/{actor_id}")
async def profile(actor_id: int) -> Dict[str, Any]:
    return {"actorId": actor_id, "preferences": load_user_preferences(actor_id), "orderProfile": java_tool_get("/tools/user-profile/" + str(actor_id))}


@app.post("/v1/knowledge")
async def add_knowledge(request: KnowledgeRequest) -> Dict[str, Any]:
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.execute("insert into knowledge_documents(title, content, embedding) values(?, ?, ?)", (request.title, request.content, json.dumps(text_embedding(request.content))))
    return {"id": cursor.lastrowid, "indexed": True}


@app.get("/v1/knowledge/search")
async def search_knowledge(q: str) -> Dict[str, Any]:
    return {"query": q, "documents": retrieve_knowledge(q)}


def get_task(task_id: str) -> Dict[str, Any]:
    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute("select task_id, actor_id, role, task_type, payload, status, result, created_at, updated_at from agent_tasks where task_id=?", (task_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="task not found")
    return {"taskId": row[0], "actorId": row[1], "role": row[2], "taskType": row[3], "payload": json.loads(row[4]), "status": row[5], "result": json.loads(row[6]), "createdAt": row[7], "updatedAt": row[8]}


def update_task(task_id: str, status: str, result: Dict[str, Any]) -> None:
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("update agent_tasks set status=?, result=?, updated_at=? where task_id=?", (status, json.dumps(result, ensure_ascii=False), int(time.time()), task_id))


async def process_agent_task(task_id: str, request: AgentRequest) -> None:
    update_task(task_id, "RUNNING", {})
    try:
        state = await agent_graph.ainvoke({"request": request})
        update_task(task_id, "COMPLETED", state["result"])
    except Exception as exc:
        update_task(task_id, "FAILED", {"error": str(exc)})


@app.post("/v1/agent/tasks")
async def create_async_task(request: AgentRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    task_id = create_task(request.actor_id, request.role, request.intent, request.model_dump(), "QUEUED")
    background_tasks.add_task(process_agent_task, task_id, request)
    return {"taskId": task_id, "status": "QUEUED"}


@app.post("/v1/agent/tasks/proposals")
async def create_proposal(request: TaskRequest) -> Dict[str, Any]:
    return {"taskId": create_task(request.actor_id, request.role, request.task_type, request.action), "status": "WAITING_CONFIRMATION"}


@app.get("/v1/agent/tasks/{task_id}")
async def task_detail(task_id: str) -> Dict[str, Any]:
    return get_task(task_id)


@app.post("/v1/agent/tasks/{task_id}/confirm")
async def confirm_task(task_id: str) -> Dict[str, Any]:
    task = get_task(task_id)
    if task["status"] != "WAITING_CONFIRMATION":
        raise HTTPException(status_code=409, detail="task is not waiting for confirmation")
    base_url = os.getenv("JAVA_TOOL_BASE_URL", "http://127.0.0.1:8080/internal/agent").rstrip("/")
    response = httpx.post(base_url + "/actions/execute", headers={"X-Agent-Token": os.getenv("JAVA_TOOL_TOKEN", "sky-agent-local-token")}, json=task["payload"], timeout=15)
    response.raise_for_status()
    result = response.json()
    if result.get("code") != 1:
        raise HTTPException(status_code=400, detail=result.get("msg") or "execution failed")
    update_task(task_id, "COMPLETED", result.get("data") or {})
    return get_task(task_id)


@app.get("/metrics")
async def agent_metrics() -> Dict[str, Any]:
    requests = metrics["requests"]
    with sqlite3.connect(DB_PATH) as connection:
        feedback_count = connection.execute("select count(*) from feedback").fetchone()[0]
        audit_count = connection.execute("select count(*) from agent_runs").fetchone()[0]
    return {**metrics, "averageLatencyMs": round(metrics["latency_ms_total"] / requests, 2) if requests else 0, "feedbackCount": feedback_count, "auditCount": audit_count}


@app.get("/prometheus")
async def prometheus_metrics() -> Response:
    requests = metrics["requests"]
    lines = [
        "# HELP sky_agent_requests_total Total Agent requests",
        "# TYPE sky_agent_requests_total counter",
        "sky_agent_requests_total " + str(requests),
        "# TYPE sky_agent_tool_success_total counter",
        "sky_agent_tool_success_total " + str(metrics["tool_success"]),
        "# TYPE sky_agent_tool_failure_total counter",
        "sky_agent_tool_failure_total " + str(metrics["tool_failure"]),
        "# TYPE sky_agent_model_failures_total counter",
        "sky_agent_model_failures_total " + str(metrics["model_failures"]),
        "# TYPE sky_agent_latency_ms gauge",
        "sky_agent_latency_ms " + str(round(metrics["latency_ms_total"] / requests, 2) if requests else 0),
    ]
    return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")
