from __future__ import annotations

import json
import time
from pathlib import Path

import cactus

from tools_schema import SPIDOR_TOOLS, tool_call_to_raid_command


WEIGHTS_PATH = Path("models/spidor_needle3_ru.cact")

weights = str(WEIGHTS_PATH) if WEIGHTS_PATH.exists() else None
print(f"Weights: {weights or 'base Needle 3'}")
agent = cactus.Cactus(tools=SPIDOR_TOOLS, weights=weights)
print("query (null string to exit):")

while True:
    try:
        query = input("\n> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not query or query.lower() in {"exit", "quit"}:
        break

    agent.reset()
    t0 = time.perf_counter()
    resp = agent.complete(query)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    text = (resp.get("response") or "").strip()
    calls = resp.get("function_calls") or resp.get("suppressed_calls") or []
    if text:
        print(f"Text:  {text}")
    if calls:
        for call in calls:
            name = str(call.get("name", ""))
            args = call.get("arguments") or {}
            print(f"Tool:  {name}({json.dumps(args, ensure_ascii=False)})")
            try:
                raid_cmd = tool_call_to_raid_command(name, args)
                print(f"Raid:  {json.dumps(raid_cmd, ensure_ascii=False)}")
            except ValueError as err:
                print(f"Raid:  {err}")
    else:
        print("Tool:  (none)")
    print(f"Time:  {dt_ms:.1f} ms")
