#!/usr/bin/env python3
"""
whisper_watcher.py — Discord #whisper listener (explicit-trigger mode).

Polls #whisper for new messages but only processes when:
  1. The message is from George (GEORGE_USER_ID).
  2. The message contains an explicit trigger keyword or dispatch JSON.

Trigger keywords: summarize, tl;dr, tldr, transcribe, process
Dispatch JSON:    {"action": "process"} (or "transcribe" / "summarize")
Dedupe override:  include "process anyway" or "retry" to bypass 24h dedupe.

Auto-watch (no explicit prompt) is DISABLED.

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

# QUIET_MODE: when this file exists, all Discord posts from this watcher are
# suppressed and in-flight jobs short-circuit before posting.
# To re-enable: delete .whisper_quiet_mode or run with --disable-quiet-mode.
QUIET_MODE_FILE = Path(__file__).parent / ".whisper_quiet_mode"


def is_quiet_mode() -> bool:
    return QUIET_MODE_FILE.exists()

# Only messages from this user will trigger processing.
GEORGE_USER_ID = "732025456617979925"

# Explicit trigger keywords (any substring match, case-insensitive).
TRIGGER_KEYWORDS = ("summarize", "tl;dr", "tldr", "transcribe", "process")

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
    if is_quiet_mode():
        print(f"[whisper-watcher] QUIET_MODE active — suppressed post: {message[:80]!r}")
        return
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


# ── Gate checks ───────────────────────────────────────────────────────────

def is_from_george(msg: dict) -> bool:
    """Return True if the message author is George. Rejects if author unknown."""
    author = msg.get("author") or {}
    return str(author.get("id", "")) == GEORGE_USER_ID


def has_trigger(content: str) -> bool:
    """Return True if content contains an explicit processing trigger or dispatch JSON."""
    lower = content.lower()
    if any(kw in lower for kw in TRIGGER_KEYWORDS):
        return True
    # Dispatch JSON: {"action": "process"} / {"action": "transcribe"} etc.
    try:
        data = json.loads(content.strip())
        return isinstance(data, dict) and data.get("action") in ("process", "transcribe", "summarize")
    except Exception:
        return False


# ── Main Loop ─────────────────────────────────────────────────────────────

def process_message(msg: dict, state: dict) -> bool:
    """Process a single message. Returns True if anything was processed.

    Note: msg_id is added to processed_ids AFTER work completes (not before).
    Fetch failures are NOT marked processed so they can be retried next poll.
    Dedupe skips ARE marked processed to prevent repeated nag messages.
    """
    msg_id = msg["id"]
    content = msg.get("content", "")
    attachments = msg.get("attachments", [])
    force = has_override(content)

    processed = False
    any_work_attempted = False

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
                any_work_attempted = True  # mark so we don't nag again
                continue

            any_work_attempted = True
            try:
                video_path = download_attachment(att["url"], tmp)
                content_hash = file_hash(video_path)
                if is_quiet_mode():
                    print(f"[whisper-watcher] QUIET_MODE active — aborting job for {att['filename']}")
                    continue
                summary = transcribe_and_summarize(video_path)
                post_to_whisper(f"**{att['filename']}**\n{summary}")
                dedupe_mark(source_id, "file", content_hash)
                processed = True
            except Exception as e:
                print(f"[whisper-watcher] Failed: {e}")
                post_to_whisper(f"⚠️ Failed to process `{att['filename']}`: {e}")
                # Do NOT mark msg_id processed — fetch failures are retryable

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
                any_work_attempted = True
                continue

            any_work_attempted = True
            try:
                video_path = download_url(url, tmp)
                if is_quiet_mode():
                    print(f"[whisper-watcher] QUIET_MODE active — aborting job for {url[:60]}")
                    continue
                summary = transcribe_and_summarize(video_path)
                short_url = url[:60] + "..." if len(url) > 60 else url
                post_to_whisper(f"**{short_url}**\n{summary}")
                dedupe_mark(source_id, "url")
                processed = True
            except Exception as e:
                print(f"[whisper-watcher] Failed: {e}")
                post_to_whisper(f"⚠️ Failed to process URL: {e}")
                # Do NOT mark msg_id processed — fetch failures are retryable

    # Only mark as seen if we did real work or hit a dedupe skip.
    # Fetch failures are left unmarked so next poll retries them.
    if processed or any_work_attempted:
        state["processed_ids"].append(msg_id)
        state["processed_ids"] = state["processed_ids"][-2000:]
        save_state(state)

    return processed


def poll_once(state: dict) -> None:
    messages = read_messages(after_id=state.get("last_message_id"))
    if not messages:
        return

    # Track latest seen ID regardless of whether we process anything
    latest_id = max(m["id"] for m in messages)
    state["last_message_id"] = latest_id

    processed_ids = set(state.get("processed_ids", []))
    for msg in sorted(messages, key=lambda m: m["id"]):
        if msg["id"] in processed_ids:
            continue

        # Gate 1: must be from George
        if not is_from_george(msg):
            continue

        has_video = any(a.get("content_type", "").startswith("video/") for a in msg.get("attachments", []))
        has_url = bool(extract_urls(msg.get("content", "")))
        if not (has_video or has_url):
            continue

        # Gate 2: must contain explicit trigger or dispatch JSON
        content = msg.get("content", "")
        if not has_trigger(content):
            print(f"[whisper-watcher] Skipping msg {msg['id']} — no explicit trigger from George")
            continue

        process_message(msg, state)

    save_state(state)


def main():
    parser = argparse.ArgumentParser(description="whisper-watcher: Discord #whisper listener (explicit-trigger mode)")
    parser.add_argument("--once", action="store_true", help="Poll once and exit")
    parser.add_argument("--disable-quiet-mode", action="store_true",
                        help="Remove the quiet-mode sentinel and resume normal posting")
    args = parser.parse_args()

    if args.disable_quiet_mode:
        if QUIET_MODE_FILE.exists():
            QUIET_MODE_FILE.unlink()
            print("[whisper-watcher] QUIET_MODE disabled — resuming normal operation.")
        else:
            print("[whisper-watcher] QUIET_MODE was not active.")

    lock_fp = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("[whisper-watcher] Another instance is already running. Exiting.")
        sys.exit(1)

    state = load_state()
    print(f"[whisper-watcher] Starting (explicit-trigger mode). Watching #whisper ({WHISPER_CHANNEL})")
    print(f"[whisper-watcher] Only processing requests from George ({GEORGE_USER_ID}) with explicit trigger.")

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
