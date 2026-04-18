from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
import whisper

_models: dict[str, whisper.Whisper] = {}
_console = Console()

MODEL_SIZES = {
    "tiny": "~75MB",
    "base": "~140MB",
    "small": "~460MB",
    "medium": "~1.5GB",
    "large": "~2.9GB",
}


@dataclass
class TranscriptResult:
    text: str
    language: str
    duration_secs: float


def _get_model(model_name: str) -> whisper.Whisper:
    if model_name not in _models:
        size = MODEL_SIZES.get(model_name, "unknown size")
        _console.print(f"[dim]Loading whisper model '{model_name}' ({size})... first run downloads it.[/dim]")
        _models[model_name] = whisper.load_model(model_name)
        _console.print(f"[dim]Model '{model_name}' ready.[/dim]")
    return _models[model_name]


def transcribe(video_path: Path, model_name: str = "base") -> TranscriptResult:
    model = _get_model(model_name)
    result = model.transcribe(str(video_path))
    text = result["text"].strip()
    language = result.get("language", "unknown")
    segments = result.get("segments", [])
    duration_secs = segments[-1]["end"] if segments else 0.0
    return TranscriptResult(text=text, language=language, duration_secs=duration_secs)
