"""Smoke test for session_id-keyed AP-Env endpoints."""
import subprocess, sys, time, requests

proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app.main:app", "--port", "7860"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
)
time.sleep(3)

BASE = "http://localhost:7860"
passed = failed = 0

def ok(label):
    global passed
    passed += 1
    print(f"  [PASS] {label}")

def fail(label, detail=""):
    global failed
    failed += 1
    print(f"  [FAIL] {label}  {detail}")

try:
    # 1 — /reset returns session_id + observation
    r = requests.post(f"{BASE}/reset", params={"task_name": "easy"}, timeout=10).json()
    sid = r.get("session_id", "")
    if sid and "observation" in r:
        ok(f"/reset  session_id={sid[:8]}...")
    else:
        fail("/reset missing session_id or observation", str(r))

    # 2 — /step accepts session_id, returns reward
    first_email_id = r["observation"]["inbox_status"][0]["id"]
    action = {"action_type": "read_email", "email_id": first_email_id}
    r2 = requests.post(f"{BASE}/step", params={"session_id": sid}, json=action, timeout=10).json()
    if "reward" in r2 and "done" in r2:
        ok(f"/step   reward={r2['reward']}  done={r2['done']}")
    else:
        fail("/step bad response", str(r2))

    # 3 — old task_name param on /step now returns 422 (session_id is required)
    r3 = requests.post(f"{BASE}/step", params={"task_name": "easy"}, json=action, timeout=10)
    if r3.status_code == 422:
        ok("/step without session_id → 422 Unprocessable")
    else:
        fail(f"/step without session_id → expected 422, got {r3.status_code}")

    # 4 — bogus session_id → 404
    r4 = requests.post(f"{BASE}/step", params={"session_id": "not-a-real-id"}, json=action, timeout=10)
    if r4.status_code == 404:
        ok("bad session_id → 404")
    else:
        fail(f"bad session_id → expected 404, got {r4.status_code}")

    # 5 — parallel sessions have distinct IDs and don't collide
    ra = requests.post(f"{BASE}/reset", params={"task_name": "easy"},   timeout=10).json()
    rb = requests.post(f"{BASE}/reset", params={"task_name": "medium"}, timeout=10).json()
    sid_a, sid_b = ra["session_id"], rb["session_id"]
    if sid_a != sid_b:
        ok(f"parallel sessions distinct  {sid_a[:8]}... vs {sid_b[:8]}...")
    else:
        fail("parallel sessions returned same session_id!")

    # 6 — /sessions shows active count
    r6 = requests.get(f"{BASE}/sessions", timeout=10).json()
    count = r6.get("active_sessions", -1)
    if count >= 2:
        ok(f"/sessions active_sessions={count}")
    else:
        fail(f"/sessions unexpected count: {r6}")

    # 7 — session eagerly deleted after done
    # Run a full easy episode to completion
    rr = requests.post(f"{BASE}/reset", params={"task_name": "easy"}, timeout=10).json()
    fsid = rr["session_id"]
    obs  = rr["observation"]
    for _ in range(35):
        act = {"action_type": "approve"}
        result = requests.post(f"{BASE}/step", params={"session_id": fsid}, json=act, timeout=10).json()
        if result.get("done"):
            break
    r7 = requests.post(f"{BASE}/step", params={"session_id": fsid}, json=act, timeout=10)
    if r7.status_code == 404:
        ok("finished session eagerly evicted → 404 on next /step")
    else:
        fail(f"finished session still alive after done  status={r7.status_code}")

    # 8 — /health still works
    r8 = requests.get(f"{BASE}/health", timeout=10).json()
    if r8.get("status") == "healthy":
        ok(f"/health  active_sessions={r8.get('active_sessions')}")
    else:
        fail("/health bad response", str(r8))

finally:
    proc.terminate()

print()
print(f"{'='*40}")
print(f"  Results: {passed} passed, {failed} failed")
print(f"{'='*40}")
sys.exit(0 if failed == 0 else 1)
