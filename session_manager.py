import json
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path

# Data dir setup
DATA_DIR = Path("./data/sessions")
DATA_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_PATH = DATA_DIR / "registry.json"


class _SafeEncoder(json.JSONEncoder):
    """Handle Pydantic models, Enums, and other non-serializable objects."""

    def default(self, obj):
        # Pydantic v2 BaseModel
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        # Pydantic v1 BaseModel
        if hasattr(obj, "dict"):
            return obj.dict()
        # Enum values
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


def _load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_registry(registry: dict):
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, cls=_SafeEncoder)

def list_sessions() -> list[dict]:
    registry = _load_registry()
    sessions = list(registry.values())
    sessions.sort(key=lambda s: s.get("last_active", ""), reverse=True)
    return sessions

def create_session(name: str) -> str:
    registry = _load_registry()
    session_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    registry[session_id] = {
        "id": session_id,
        "name": name,
        "created_at": now,
        "last_active": now,
        "queries_made": 0,
        "papers_retrieved": 0,
        "chat_history": []
    }
    _save_registry(registry)
    return session_id

def load_session(session_id: str) -> dict:
    registry = _load_registry()
    return registry.get(session_id)

def save_chat_history(session_id: str, history: list):
    registry = _load_registry()
    if session_id in registry:
        registry[session_id]["chat_history"] = history
        registry[session_id]["last_active"] = datetime.now().isoformat()
        _save_registry(registry)

def update_session_stats(session_id: str, new_papers_count: int):
    registry = _load_registry()
    if session_id in registry:
        registry[session_id]["queries_made"] += 1
        registry[session_id]["papers_retrieved"] += new_papers_count
        registry[session_id]["last_active"] = datetime.now().isoformat()
        _save_registry(registry)

def delete_session(session_id: str):
    registry = _load_registry()
    if session_id in registry:
        del registry[session_id]
        _save_registry(registry)

def export_session(session_id: str) -> str:
    session = load_session(session_id)
    if not session:
        return "{}"
    return json.dumps(session, indent=2, cls=_SafeEncoder)
