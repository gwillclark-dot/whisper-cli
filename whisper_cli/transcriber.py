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


def _get_model(model_name: str) -> whisper.Whisper:
    if model_name not in _models:
        size = MODEL_SIZES.get(model_name, "unknown size")
        _console.print(f"[dim]Loading whisper model '{model_name}' ({size})... first run downloads it.[/dim]")
        _models[model_name] = whisper.load_model(model_name)
        _console.print(f"[dim]Model '{model_name}' ready.[/dim]")
    return _models[model_name]


def transcribe(video_path: Path, model_name: str = "base") -> str:
    model = _get_model(model_name)
    result = model.transcribe(str(video_path))
    return result["text"].strip()
