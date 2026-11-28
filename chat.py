from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

try:
    import readline  # noqa: F401
except ImportError:
    pass

try:
    import termios

    if sys.stdin.isatty():
        _attrs = termios.tcgetattr(sys.stdin.fileno())
        if hasattr(termios, "IUTF8"):
            _attrs[0] |= termios.IUTF8
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, _attrs)
except Exception:
    pass

import cactus

from offtopic_filter import CLF_PATH, OfftopicFilter
from tools_schema import SPIDOR_TOOLS, tool_call_to_raid_command

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|[\x00-\x1f\x7f]")


WEIGHTS_PATH = Path("models/spidor_needle3_ru.cact")

weights = str(WEIGHTS_PATH) if WEIGHTS_PATH.exists() else None
offtopic_filter = OfftopicFilter() if CLF_PATH.exists() else None
print(f"Weights: {weights or 'base Needle 3'} | Layer 1: {'rubert-tiny2' if offtopic_filter else 'off'}")
agent = cactus.Cactus(tools=SPIDOR_TOOLS, weights=weights)
print("query (null string to exit):")

while True:
    try:
        raw_in = input("\n> ").encode("utf-8", "ignore").decode("utf-8")
        query = ANSI_ESCAPE_RE.sub("", raw_in).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not query or query.lower() in {"exit", "quit"}:
        break

    t0 = time.perf_counter()
    if offtopic_filter is not None:
        calls, src = offtopic_filter.route(query, agent)
    else:
        agent.reset()
        resp = agent.complete(query)
        calls = resp.get("function_calls") or resp.get("suppressed_calls") or []
        src = "needle3"
    dt_ms = (time.perf_counter() - t0) * 1000.0

    if calls:
        for call in calls:
            name = str(call.get("name", ""))
            args = call.get("arguments") or {}
            print(f"Tool:  {name}({json.dumps(args, ensure_ascii=False)}) [{src}]")
            try:
                raid_cmd = tool_call_to_raid_command(name, args)
                print(f"Raid:  {json.dumps(raid_cmd, ensure_ascii=False)}")
            except ValueError as err:
                print(f"Raid:  {err}")
    else:
        print(f"Tool:  (none) [{src}]")
    print(f"Time:  {dt_ms:.1f} ms")
