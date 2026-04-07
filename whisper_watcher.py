#!/usr/bin/env python3
"""
whisper_watcher.py — Discord #whisper listener.

Polls #whisper for new messages with video attachments or YouTube/TikTok URLs.
Downloads each, transcribes via Whisper API, summarizes via GPT-4o-mini,
and posts the result back to #whisper.

Run manually:  python3 whisper_watcher.py
Run once:      python3 whisper_watcher.py --once
"""

import argparse
import fcntl
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from whisper_cli.dedupe import file_hash, has_override, is_duplicate, mark_processed as dedupe_mark

# ── Config ────────────────────────────────────────────────────────────────

WHISPER_CHANNEL = "1490901110414905580"
STATE_FILE = Path(__file__).parent / ".whisper_watcher_state.json"
LOCK_FILE = Path(__file__).parent / ".whisper_watcher.lock"
POLL_INTERVAL = 30  # seconds

# ── State ─────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"processed_ids": [], "last_message_id": None}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── Discord ───────────────────────────────────────────────────────────────

def read_messages(after_id: str | None = None) -> list[dict]:
    cmd = ["clawdbot", "message", "read", "--channel", "discord",
           "--target", WHISPER_CHANNEL, "--json", "--limit", "20"]
    if after_id:
        cmd += ["--after", after_id]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[whisper-watcher] clawdbot read failed: {result.stderr.strip()}")
        return []
    try:
        data = json.loads(result.stdout)
        return data["payload"]["messages"]
    except Exception as e:
        print(f"[whisper-watcher] parse error: {e}")
        return []


def post_to_whisper(message: str) -> None:
    subprocess.run(
        ["clawdbot", "message", "send", "--channel", "discord",
         "--target", WHISPER_CHANNEL, "--message", message],
        capture_output=True,
    )


# ── Download ──────────────────────────────────────────────────────────────

def download_attachment(url: str, dest_dir: Path) -> Path:
    """Download a Discord CDN attachment via curl."""
    filename = url.split("/")[-1].split("?")[0]
    dest = dest_dir / filename
    result = subprocess.run(
        ["curl", "-L", "-s", "-o", str(dest), url],
        capture_output=True,
    )
    if result.returncode != 0 or not dest.exists():
        raise RuntimeError(f"curl failed for {url}")
    return dest


def download_url(url: str, dest_dir: Path) -> Path:
    """Download YouTube/TikTok via yt-dlp."""
    template = str(dest_dir / "%(title).60s.%(ext)s")
    result = subprocess.run(
        ["yt-dlp", "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
         "--merge-output-format", "mp4", "--output", template, "--no-playlist",
         "--print", "after_move:filepath", url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr.strip()}")
    filepath = result.stdout.strip().splitlines()[-1]
    path = Path(filepath)
    if not path.exists():
        files = sorted(dest_dir.glob("*.mp4"), key=lambda f: f.stat().st_mtime)
        if not files:
            raise RuntimeError("yt-dlp succeeded but no mp4 found")
        path = files[-1]
    return path


# ── Transcribe + Summarize ────────────────────────────────────────────────

def transcribe_and_summarize(video_path: Path) -> str:
    """Run transcription + summarization, return formatted result."""
    from whisper_cli.config import load_config
    from whisper_cli.transcriber import transcribe
    from whisper_cli.summarizer import summarize

    cfg = load_config()

    print(f"[whisper-watcher] Transcribing {video_path.name}...")
    transcript = transcribe(video_path, api_key=cfg.openai_api_key)

    print(f"[whisper-watcher] Summarizing ({len(transcript)} chars)...")
    summary = summarize(transcript, video_path.name, cfg.openai_api_key)

    return summary


# ── URL Detection ─────────────────────────────────────────────────────────

SUPPORTED_DOMAINS = ("youtube.com", "youtu.be", "tiktok.com", "vm.tiktok.com")


def extract_urls(content: str) -> list[str]:
    """Pull YouTube/TikTok URLs from message content."""
    urls = []
    for word in content.split():
        if word.startswith("http") and any(d in word for d in SUPPORTED_DOMAINS):
            urls.append(word.strip("<>"))
    return urls


# ── Main Loop ─────────────────────────────────────────────────────────────

def process_message(msg: dict, state: dict) -> bool:
    """Process a single message. Returns True if anything was processed."""
    msg_id = msg["id"]
    content = msg.get("content", "")
    attachments = msg.get("attachments", [])
    force = has_override(content)

    # Mark processed BEFORE doing work to prevent duplicate posts on crash/retry
    state["processed_ids"].append(msg_id)
    state["processed_ids"] = state["processed_ids"][-2000:]
    save_state(state)

    processed = False

    with tempfile.TemporaryDirectory(prefix="whisper_dl_") as tmpdir:
        tmp = Path(tmpdir)

        # Process video attachments
        for att in attachments:
            if not att.get("content_type", "").startswith("video/"):
                continue
            source_id = att["filename"]
            print(f"[whisper-watcher] Attachment: {source_id} ({att['size']//1024}KB)")

            if not force and is_duplicate(source_id, "file"):
                print(f"[whisper-watcher] Dedupe: {source_id} already processed within 24h")
                post_to_whisper(
                    f"⚠️ Looks like the last item (`{source_id}`) is the same as this one. "
                    "Re-run? Reply: process anyway"
                )
                continue

            try:
                video_path = download_attachment(att["url"], tmp)
                content_hash = file_hash(video_path)
                summary = transcribe_and_summarize(video_path)
                post_to_whisper(f"**{att['filename']}**\n{summary}")
                dedupe_mark(source_id, "file", content_hash)
                processed = True
            except Exception as e:
                print(f"[whisper-watcher] Failed: {e}")
                post_to_whisper(f"⚠️ Failed to process `{att['filename']}`: {e}")

        # Process YouTube/TikTok URLs
        for url in extract_urls(content):
            source_id = url.rstrip("/")
            print(f"[whisper-watcher] URL: {source_id}")

            if not force and is_duplicate(source_id, "url"):
                print(f"[whisper-watcher] Dedupe: URL already processed within 24h")
                short_url = source_id[:60] + "..." if len(source_id) > 60 else source_id
                post_to_whisper(
                    f"⚠️ Looks like the last item (`{short_url}`) is the same as this one. "
                    "Re-run? Reply: process anyway"
                )
                continue

            try:
                video_path = download_url(url, tmp)
                summary = transcribe_and_summarize(video_path)
                short_url = url[:60] + "..." if len(url) > 60 else url
                post_to_whisper(f"**{short_url}**\n{summary}")
                dedupe_mark(source_id, "url")
                processed = True
            except Exception as e:
                print(f"[whisper-watcher] Failed: {e}")
                post_to_whisper(f"⚠️ Failed to process URL: {e}")

    return processed


def poll_once(state: dict) -> None:
    messages = read_messages(after_id=state.get("last_message_id"))
    if not messages:
        return

    # Track latest seen ID
    latest_id = max(m["id"] for m in messages)
    state["last_message_id"] = latest_id

    # Filter to unprocessed messages that have actionable content
    processed_ids = set(state.get("processed_ids", []))
    for msg in sorted(messages, key=lambda m: m["id"]):
        if msg["id"] in processed_ids:
            continue
        has_video = any(a.get("content_type", "").startswith("video/") for a in msg.get("attachments", []))
        has_url = bool(extract_urls(msg.get("content", "")))
        if has_video or has_url:
            process_message(msg, state)

    save_state(state)


def main():
    parser = argparse.ArgumentParser(description="whisper-watcher: Discord #whisper listener")
    parser.add_argument("--once", action="store_true", help="Poll once and exit")
    args = parser.parse_args()

    lock_fp = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("[whisper-watcher] Another instance is already running. Exiting.")
        sys.exit(1)

    state = load_state()
    print(f"[whisper-watcher] Starting. Watching #whisper ({WHISPER_CHANNEL})")

    if args.once:
        poll_once(state)
        return

    try:
        while True:
            poll_once(state)
            print(f"[whisper-watcher] Sleeping {POLL_INTERVAL}s...")
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print("\n[whisper-watcher] Stopped.")
    finally:
        fcntl.flock(lock_fp, fcntl.LOCK_UN)
        lock_fp.close()


if __name__ == "__main__":
    main()
