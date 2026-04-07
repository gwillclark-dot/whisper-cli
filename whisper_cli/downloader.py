"""Download audio/video from YouTube or TikTok URLs using yt-dlp."""

import subprocess
import shutil
from pathlib import Path


SUPPORTED_DOMAINS = ("youtube.com", "youtu.be", "tiktok.com", "vm.tiktok.com")


def is_supported_url(url: str) -> bool:
    return any(domain in url for domain in SUPPORTED_DOMAINS)


def check_ytdlp() -> None:
    if not shutil.which("yt-dlp"):
        raise SystemExit(
            "yt-dlp is required but not found.\n"
            "  Install: pip install yt-dlp  or  brew install yt-dlp"
        )


def download(url: str, dest_dir: Path) -> Path:
    """Download URL to dest_dir, return path to downloaded file (mp4)."""
    check_ytdlp()
    dest_dir.mkdir(parents=True, exist_ok=True)

    output_template = str(dest_dir / "%(title).60s.%(ext)s")

    result = subprocess.run(
        [
            "yt-dlp",
            "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "--output", output_template,
            "--no-playlist",
            "--print", "after_move:filepath",
            url,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr.strip()}")

    # Last printed line is the filepath
    filepath = result.stdout.strip().splitlines()[-1]
    path = Path(filepath)
    if not path.exists():
        # Fallback: find the most recently created mp4 in dest_dir
        files = sorted(dest_dir.glob("*.mp4"), key=lambda f: f.stat().st_mtime)
        if not files:
            raise RuntimeError(f"yt-dlp succeeded but no mp4 found in {dest_dir}")
        path = files[-1]

    return path
