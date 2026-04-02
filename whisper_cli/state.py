import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from whisper_cli.scanner import VideoFile


@dataclass
class ProcessedEntry:
    mtime: float
    size_bytes: int
    processed_at: str
    status: str
    transcript_chars: int = 0
    summary_chars: int = 0
    error: str | None = None


@dataclass
class State:
    version: int = 1
    processed: dict[str, ProcessedEntry] = field(default_factory=dict)


def output_base(folder: Path, override: Path | None = None) -> Path:
    """Return the output directory. Default: <folder>/vidsum-output/"""
    base = override if override else folder / "vidsum-output"
    base.mkdir(parents=True, exist_ok=True)
    return base


def state_path(base: Path) -> Path:
    return base / "state.json"


def load_state(path: Path) -> State:
    if not path.exists():
        return State()
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        backup = path.with_suffix(".json.bak")
        path.rename(backup)
        return State()
    state = State(version=data.get("version", 1))
    for key, entry in data.get("processed", {}).items():
        state.processed[key] = ProcessedEntry(**entry)
    return state


def save_state(state: State, path: Path) -> None:
    data = {"version": state.version, "processed": {k: asdict(v) for k, v in state.processed.items()}}
    path.write_text(json.dumps(data, indent=2) + "\n")


def get_unprocessed(videos: list[VideoFile], state: State) -> list[VideoFile]:
    unprocessed = []
    for v in videos:
        key = str(v.path)
        entry = state.processed.get(key)
        if entry is None or entry.mtime != v.mtime or entry.status.startswith("error"):
            unprocessed.append(v)
    return unprocessed


def mark_processed(
    state: State,
    video: VideoFile,
    status: str,
    transcript_chars: int = 0,
    summary_chars: int = 0,
    error: str | None = None,
) -> None:
    state.processed[str(video.path)] = ProcessedEntry(
        mtime=video.mtime,
        size_bytes=video.size_bytes,
        processed_at=datetime.now().isoformat(timespec="seconds"),
        status=status,
        transcript_chars=transcript_chars,
        summary_chars=summary_chars,
        error=error,
    )
