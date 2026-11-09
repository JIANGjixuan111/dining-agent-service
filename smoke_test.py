import json
import urllib.request

BASE_URL = "http://127.0.0.1:8001"


def request(path, method="GET", body=None):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body else None
    req = urllib.request.Request(BASE_URL + path, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


health = request("/health")
assert health["status"] == "ok"
assert health["graph"] == ["prepare", "planner", "tool_executor", "response_builder"]

result = request("/v1/agent/chat", "POST", {
    "role": "user", "actor_id": 1, "intent": "plan_order",
    "message": "两个人预算80元，请生成完整方案",
    "context": {"dishes": [
        {"id": 1, "name": "米饭", "price": 3},
        {"id": 2, "name": "清炒西兰花", "price": 18},
        {"id": 3, "name": "番茄鸡蛋汤", "price": 22},
        {"id": 4, "name": "烤鱼", "price": 88}
    ]}
})
assert result["toolCalls"][0]["name"] == "build_budget_plan"
assert result["action"]["type"] == "CONFIRM_CART"
assert result["toolResults"]["build_budget_plan"]["budget"] == 80

session = request("/v1/agent/sessions/" + result["sessionId"])
assert len(session["history"]) == 2

feedback = request("/v1/agent/feedback", "POST", {"actor_id": 1, "session_id": result["sessionId"], "rating": 1, "comment": "smoke test"})
assert feedback["accepted"] is True

print(json.dumps({"health": health, "sessionId": result["sessionId"], "tools": result["toolCalls"], "suggestions": len(result["suggestions"])}, ensure_ascii=False))
