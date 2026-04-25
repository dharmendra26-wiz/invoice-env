import time
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException
from typing import Dict, Any

from app.environment import EnterpriseAPEnvironment
from app.models import Action, ResetResponse, StepResult

app = FastAPI(title="Enterprise AP Environment", version="2.0.0")

# ── Session store ─────────────────────────────────────────────────────────────
# Keyed by UUID session_id so parallel episodes never collide.
_envs: Dict[str, EnterpriseAPEnvironment] = {}
_last_active: Dict[str, float] = {}   # session_id → monotonic timestamp
SESSION_TTL = 300.0                    # seconds before an idle session is evicted


def _evict_stale() -> None:
    """Remove sessions that have been idle longer than SESSION_TTL."""
    cutoff = time.monotonic() - SESSION_TTL
    stale = [sid for sid, t in _last_active.items() if t < cutoff]
    for sid in stale:
        _envs.pop(sid, None)
        _last_active.pop(sid, None)


def _get_env(session_id: str) -> EnterpriseAPEnvironment:
    """Look up a session, refreshing its TTL.  Raises 404 if not found."""
    _evict_stale()
    if session_id not in _envs:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found or expired. Call /reset first.",
        )
    _last_active[session_id] = time.monotonic()
    return _envs[session_id]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"name": "enterprise-ap-env", "version": "2.0.0", "status": "running"}


@app.post("/reset", response_model=ResetResponse)
def reset(task_name: str = "easy") -> ResetResponse:
    """Start a new episode.  Returns a unique session_id — pass it to /step."""
    _evict_stale()
    session_id = str(uuid.uuid4())
    env = EnterpriseAPEnvironment(task_name=task_name)
    obs = env.reset()
    _envs[session_id] = env
    _last_active[session_id] = time.monotonic()
    return ResetResponse(session_id=session_id, observation=obs)


@app.post("/step", response_model=StepResult)
def step(session_id: str, action: Action) -> StepResult:
    """Advance the episode by one action.  Eagerly cleans up finished sessions."""
    env = _get_env(session_id)
    result = env.step(action)
    if result.done:
        # Episode over — free memory immediately rather than waiting for TTL.
        _envs.pop(session_id, None)
        _last_active.pop(session_id, None)
    return result


@app.get("/state")
def state(session_id: str) -> Dict[str, Any]:
    """Inspect the live state of an active session (debugging / evaluation)."""
    return _get_env(session_id).state


@app.get("/sessions")
def list_sessions() -> Dict[str, Any]:
    """Return the number of active sessions and their task names."""
    _evict_stale()
    return {
        "active_sessions": len(_envs),
        "sessions": {
            sid: env.task_name for sid, env in _envs.items()
        },
    }


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"name": "easy",               "description": "Extract fields from clean invoice and match PO"},
            {"name": "medium",             "description": "Detect line item price mismatch"},
            {"name": "hard",               "description": "Detect duplicate invoice and tax miscalculation"},
            {"name": "expert_negotiation", "description": "Email vendor to obtain corrected invoice, then approve"},
            {"name": "expert_fraud",       "description": "Detect lookalike-domain phishing invoice"},
        ]
    }


@app.get("/health")
def health():
    _evict_stale()
    return {"status": "healthy", "active_sessions": len(_envs)}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, reload=False)