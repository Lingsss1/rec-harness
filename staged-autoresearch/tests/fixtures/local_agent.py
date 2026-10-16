#!/usr/bin/env python3
"""SYNTHETIC native CLI fixture. No network/model calls; never use as a baseline."""
import json
import os
import re
import sys
from pathlib import Path

prompt = sys.stdin.read()
packet = json.loads(Path(re.search(r"Read (.+?/packet.json)", prompt).group(1)).read_text())
session_flag = "--resume" if "--resume" in sys.argv else "--session-id"
session = sys.argv[sys.argv.index(session_flag) + 1]
home = Path(os.environ["HOME"])
home.mkdir(parents=True, exist_ok=True)
seen = home / "fixture-session"
if session_flag == "--resume":
    assert seen.read_text() == session
else:
    assert not seen.exists(), "fresh context reused an old home"
    seen.write_text(session)
request = packet["request"]
role = re.search(r"ROLE=(\w+)", prompt).group(1)
direction = {"bottleneck": "SYNTHETIC", "mechanism": "SYNTHETIC"}
if role == "plan":
    result = {"direction": request["expected_direction"] or direction, "hypothesis": "SYNTHETIC",
              "falsification": "SYNTHETIC", "observables": ["SYNTHETIC"], "revision_reason": "Initial",
              "steps": [{"slot": i, "instruction": "SYNTHETIC", "condition": "always",
                         "expected_observation": "SYNTHETIC"} for i in range(1, 11)]}
elif role == "review":
    result = {"facts": [{"claim": "SYNTHETIC", "trial_refs": [0]}],
              "interpretations": [{"explanation": "SYNTHETIC", "trial_refs": [0]}],
              "uncertainties": ["SYNTHETIC"], "decision": "continue",
              "next_direction": request["current_direction"], "reason_refs": [0],
              "diagnostics": [{"kind": "paired", "parent_trial": 0, "paired_with": 0,
                 "instruction": "SYNTHETIC", "uncertainty": "SYNTHETIC",
                 "decision_if_supported": "continue", "decision_if_refuted": "switch"}]
                 if request["diagnostics_allowed"] else []}
else:
    for name in packet["task"]["editable"]:
        Path(name).write_text(str(request["trial"]))
    result = {"hypothesis": "SYNTHETIC", "change_description": "SYNTHETIC",
              "expected_observation": "SYNTHETIC"}
print(json.dumps({"session_id": session, "structured_output": result, "is_error": False,
                  "usage": {"input_tokens": 1, "output_tokens": 1}, "total_cost_usd": 0}))
