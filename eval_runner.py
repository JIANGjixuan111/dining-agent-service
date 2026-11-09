import json
import os
import sys

import httpx

BASE_URL = os.getenv("AGENT_EVAL_URL", "http://127.0.0.1:8001")

with open(os.path.join(os.path.dirname(__file__), "eval_cases.json"), encoding="utf-8") as source:
    cases = json.load(source)

passed = 0
for case in cases:
    response = httpx.post(BASE_URL + "/v1/agent/chat", json={
        "role": case["role"], "actor_id": 1, "intent": case["intent"],
        "message": case["message"], "context": {},
    }, timeout=120)
    response.raise_for_status()
    result = response.json()
    actual = {item["name"] for item in result.get("toolCalls", []) if item.get("status") == "success"}
    expected = set(case["required_tools"])
    ok = expected.issubset(actual)
    passed += int(ok)
    print(("PASS" if ok else "FAIL") + " " + case["name"] + " tools=" + ",".join(sorted(actual)))

print("score={}/{}".format(passed, len(cases)))
sys.exit(0 if passed == len(cases) else 1)
