#!/usr/bin/env python3
"""Dispatch: concise brief for 8b180a14 clip — uses OpenAI Whisper API, no full transcript posted."""

import os
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from whisper_cli.config import load_config
from openai import OpenAI

MEDIA = Path("/Users/g2/.clawdbot/media/inbound/8b180a14-aa0d-4c37-bcfb-ed3a863c0f01.mp4")
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


def main():
    cfg = load_config()

    if not MEDIA.exists():
        print(f"ERROR: media not found: {MEDIA}")
        sys.exit(1)

    client = OpenAI(api_key=cfg.openai_api_key)

    print("Transcribing via OpenAI API...", flush=True)
    with open(MEDIA, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    language = transcript.language or "unknown"
    segments = transcript.segments or []

    duration_secs = segments[-1].end if segments else (transcript.duration or 0)
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

    discord_msg = f"🎙 **8b180a14** | `{language}` | `{duration_fmt}`\n\n{brief}"

    print("\n--- Discord message ---")
    print(discord_msg)
    print("\n--- Posting ---", flush=True)

    subprocess.run(
        [
            "clawdbot", "message", "send",
            "--channel", "discord",
            "--target", CHANNEL,
            "--message", discord_msg,
        ],
        check=True,
    )

    print("Done.")


if __name__ == "__main__":
    main()
