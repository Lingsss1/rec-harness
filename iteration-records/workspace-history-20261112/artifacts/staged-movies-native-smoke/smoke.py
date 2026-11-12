"""Real CLI-only smoke: native edit, same-thread memory, and fresh thread."""
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "staged-autoresearch"))
from staged_research.codex import CodexRuntime
from staged_research.contracts import obj, TEXT
from staged_research.storage import read, write

root = Path(__file__).resolve().parent / "attempt-3"
work, evidence = root / "work", root / "evidence"
work.mkdir(parents=True, exist_ok=True)
evidence.mkdir(exist_ok=True)
(work / "model.py").write_text("VALUE = 1\n")
runtime = CodexRuntime({"binary": "/root/.nvm/versions/node/v24.22.0/bin/codex",
    "model": "gpt-6.1-sol", "effort": "low", "timeout_seconds": 300},
    {"isolation": "process", "agent_env_passthrough": []}, root)
schema = obj({"answer": TEXT})
session = str(uuid.uuid4())
word = "STAGED_" + uuid.uuid4().hex[:16]
common = dict(work=work, evidence=evidence, schema=schema)
first = runtime.call_agent(**common, session_home=root / "sessions/one", session=session,
    resume=False, role="experiment", receipt_dir=root / "calls/one",
    prompt=f"Using native file editing, change ONLY {work}/model.py to VALUE = 2. Do not run any scripts or tests. "
           f"Remember the codeword {word} in conversation only; do not write it to any file. Return answer READY.")
assert (work / "model.py").read_text().strip() == "VALUE = 2"
second = runtime.call_agent(**common, session_home=root / "sessions/one", session=session,
    resume=True, role="review", receipt_dir=root / "calls/two",
    prompt="Without using any tools or editing files, return the exact codeword from the previous turn as answer.")
assert second["answer"] == word, second
fresh = str(uuid.uuid4())
third = runtime.call_agent(**common, session_home=root / "sessions/two", session=fresh,
    resume=False, role="review", receipt_dir=root / "calls/three",
    prompt="Without using tools, return the codeword memorized earlier in THIS conversation. If none was provided in this conversation return UNKNOWN.")
assert third["answer"] == "UNKNOWN", third
one, two = read(root / "sessions/one/native-session.json"), read(root / "sessions/two/native-session.json")
assert one["thread_id"] != two["thread_id"]
write(root / "acceptance.json", {"native_edit": True, "native_resume_memory": True,
    "fresh_thread_has_no_prior_codeword": True, "first_thread": one["thread_id"],
    "fresh_thread": two["thread_id"], "first_completed_calls": one["completed_calls"]})
print("NATIVE_CODEX_EDIT_RESUME_FLUSH_PASSED", flush=True)
