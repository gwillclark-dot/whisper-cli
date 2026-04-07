#!/usr/bin/env python3
"""Fetch a TikTok URL, transcribe via OpenAI Whisper API, post TL;DR to Discord."""

import os
import sys
import subprocess
import tempfile
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from whisper_cli.config import load_config
from openai import OpenAI

# yt-dlp lives in the venv
YTDLP = str(REPO_ROOT / ".venv" / "bin" / "yt-dlp")

URL = "https://www.tiktok.com/t/ZP8g8N87b/"
CHANNEL = "1490901110414905580"

BRIEF_PROMPT = (
    "You are given a timestamped video transcript. Produce a concise brief — NO full transcript.\n\n"
    "Output format (use exactly these headers):\n"
    "**TL;DR**\n"
    "- 3-6 bullets. Lead with direct quotes where possible.\n"
    "- Flag action items with 🔲\n\n"
    "**Key Quotes** (2-5 short quotes with timestamps)\n"
    "- [MM:SS] \"quote\" — brief context if needed\n"
    "- Mark uncertain transcription with [?]\n\n"
    "Rules: max 300 words total. No intro or conclusion sentences."
)


def fmt_time(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def download_tiktok(url: str, out_dir: Path) -> Path:
    """Download TikTok video using yt-dlp with fresh cache and browser headers."""
    out_template = str(out_dir / "video.%(ext)s")
    cache_dir = str(out_dir / f"ytdlp-cache-{uuid.uuid4().hex[:8]}")

    cmd = [
        YTDLP,
        "--no-playlist",
        "--cache-dir", cache_dir,
        "--add-header", "Referer:https://www.tiktok.com/",
        "--add-header", "User-Agent:Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "-f", "bestaudio/best",
        "-o", out_template,
        url,
    ]

    print(f"Downloading: {url}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        # Try vm.tiktok.com variant
        vm_url = url.replace("www.tiktok.com/t/", "vm.tiktok.com/")
        print(f"First attempt failed, trying: {vm_url}", flush=True)
        cmd[-1] = vm_url
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"yt-dlp stderr:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"yt-dlp failed (exit {result.returncode}). Blocker: TikTok may require cookies.\nSuggested fix: yt-dlp --cookies-from-browser chrome {url}")

    # Find downloaded file
    candidates = sorted(out_dir.glob("video.*"))
    candidates = [f for f in candidates if f.suffix not in (".json", ".part") and "ytdlp-cache" not in str(f)]
    if not candidates:
        raise RuntimeError("yt-dlp succeeded but no output file found")
    return candidates[0]


def main():
    cfg = load_config()
    client = OpenAI(api_key=cfg.openai_api_key)

    with tempfile.TemporaryDirectory(prefix="whisper_tiktok_") as tmpdir:
        tmp = Path(tmpdir)

        # Download
        try:
            media = download_tiktok(URL, tmp)
        except RuntimeError as e:
            msg = f"⚠️ whisper/tiktok blocked\n{e}"
            print(msg)
            subprocess.run(
                ["clawdbot", "message", "send", "--channel", "discord",
                 "--target", CHANNEL, "--message", msg],
                check=False,
            )
            sys.exit(1)

        print(f"Downloaded: {media.name} ({media.stat().st_size // 1024}KB)", flush=True)

        # Transcribe via OpenAI Whisper API
        print("Transcribing...", flush=True)
        with open(media, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

        language = transcript.language or "unknown"
        segments = transcript.segments or []
        duration_secs = segments[-1].end if segments else (getattr(transcript, "duration", 0) or 0)
        duration_fmt = fmt_time(duration_secs)

        timed_lines = "\n".join(
            f"[{fmt_time(s.start)}] {s.text.strip()}"
            for s in segments
        )

        print(f"Language: {language} | Duration: {duration_fmt}", flush=True)
        print("Generating brief...", flush=True)

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": BRIEF_PROMPT},
                {"role": "user", "content": (
                    f"Language: {language} | Duration: {duration_fmt}\n\n"
                    f"Transcript:\n{timed_lines}"
                )},
            ],
            temperature=0.3,
            max_tokens=600,
        )
        brief = resp.choices[0].message.content.strip()

    slug = URL.rstrip("/").split("/")[-1]
    discord_msg = f"🎙 **TikTok/{slug}** | `{language}` | `{duration_fmt}`\n\n{brief}"

    print("\n--- Discord message ---")
    print(discord_msg)
    print("\n--- Posting ---", flush=True)

    subprocess.run(
        ["clawdbot", "message", "send",
         "--channel", "discord",
         "--target", CHANNEL,
         "--message", discord_msg],
        check=True,
    )

    print("Done.")


if __name__ == "__main__":
    main()
