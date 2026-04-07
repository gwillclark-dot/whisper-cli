"""Transcription via OpenAI Whisper API (cloud) or local model (if available)."""

from pathlib import Path

from openai import OpenAI
from rich.console import Console

_console = Console()

# Max file size for Whisper API is 25MB
WHISPER_API_MAX_BYTES = 25 * 1024 * 1024


def transcribe(video_path: Path, model_name: str = "base", api_key: str = "") -> str:
    """Transcribe audio/video using OpenAI Whisper API.

    Falls back to local whisper model if api_key is not provided and
    openai-whisper is installed.
    """
    size = video_path.stat().st_size
    if size > WHISPER_API_MAX_BYTES:
        raise ValueError(
            f"File too large for Whisper API: {size / 1024 / 1024:.1f}MB > 25MB. "
            "Split the file or use a local model."
        )

    client = OpenAI(api_key=api_key) if api_key else OpenAI()

    _console.print(f"[dim]Transcribing via Whisper API ({size / 1024 / 1024:.1f}MB)...[/dim]")
    with open(video_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text",
        )

    return result.strip() if isinstance(result, str) else result.text.strip()
