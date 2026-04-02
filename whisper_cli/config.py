import shutil
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
import os


@dataclass
class Config:
    openai_api_key: str
    snippety_csv_path: Path | None = None
    whisper_model: str = "base"
    poll_interval: int = 30


def load_config(whisper_model: str = "base", poll_interval: int = 30) -> Config:
    # Shell env takes priority, then .env file
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path, override=False)

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "your-key-here":
        raise SystemExit(
            "OPENAI_API_KEY not set.\n"
            "  Option 1: export OPENAI_API_KEY=sk-...\n"
            "  Option 2: add it to whisper-cli/.env"
        )

    snippety_path = os.getenv("SNIPPETY_CSV_PATH", "")
    snippety_csv = Path(snippety_path).expanduser() if snippety_path else None

    return Config(
        openai_api_key=api_key,
        snippety_csv_path=snippety_csv,
        whisper_model=whisper_model,
        poll_interval=poll_interval,
    )


def check_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise SystemExit(
            "ffmpeg is required but not found.\n"
            "  Install: brew install ffmpeg"
        )
