"""
dedupe.py — Last-processed dedupe guard.

Tracks the single most recent item processed. If the same source is submitted
again within WINDOW_HOURS, processing is skipped and a prompt is posted to
Discord asking for confirmation.

Override by including "process anyway" (case-insensitive) in the Discord message.
State is persisted to .whisper_state/last_processed.json (git-ignored).
"""

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

STATE_DIR = Path(__file__).parent.parent / ".whisper_state"
LAST_PROCESSED_FILE = STATE_DIR / "last_processed.json"
WINDOW_HOURS = 24
OVERRIDE_PHRASE = "process anyway"


def _load() -> dict:
    if LAST_PROCESSED_FILE.exists():
        try:
            return json.loads(LAST_PROCESSED_FILE.read_text())
        except Exception:
            pass
    return {}


def _save(data: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LAST_PROCESSED_FILE.write_text(json.dumps(data, indent=2))


def file_hash(path: Path) -> str:
    """SHA-256 of first 4 MB — fast fingerprint without reading huge files."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(4 * 1024 * 1024))
    return h.hexdigest()[:16]


def is_duplicate(source_id: str, source_type: str) -> bool:
    """Return True if source_id matches last_processed within WINDOW_HOURS."""
    last = _load()
    if not last:
        return False
    if last.get("id") != source_id or last.get("source_type") != source_type:
        return False
    processed_at = last.get("processed_at")
    if not processed_at:
        return False
    try:
        then = datetime.fromisoformat(processed_at)
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        now = datetime.now(tz=timezone.utc)
        return (now - then) < timedelta(hours=WINDOW_HOURS)
    except Exception:
        return False


def mark_processed(source_id: str, source_type: str, content_hash: Optional[str] = None) -> None:
    """Record a successful processing. Call ONLY after summary is posted."""
    _save({
        "source_type": source_type,
        "id": source_id,
        "content_hash": content_hash,
        "processed_at": datetime.now(tz=timezone.utc).isoformat(),
    })


def has_override(message_content: str) -> bool:
    """Return True if message contains the override phrase."""
    return OVERRIDE_PHRASE in message_content.lower()
