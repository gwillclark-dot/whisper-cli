#!/usr/bin/env python3
"""
whisper_watcher.py — Discord #whisper listener (explicit-trigger mode).

Polls #whisper for new messages but only processes when:
  1. The message is from George (GEORGE_USER_ID).
  2. The message contains an explicit trigger keyword or dispatch JSON.

Trigger keywords: summarize, tl;dr, tldr, transcribe, process
Dispatch JSON:    {"action": "process"} (or "transcribe" / "summarize")
Dedupe override:  include "process anyway" or "retry" to bypass 24h dedupe.

Per-source locking and 120s debounce prevent duplicate jobs.

Single-message lifecycle:
  - Post one "⏳ Working…" placeholder (skipped in QUIET_MODE)
  - Edit that message with the final TL;DR or error
  - Retries reuse the same Discord message (no new posts)

QUIET_MODE: when .whisper_quiet_mode exists, suppress the "Working…" placeholder
  and any intermediate posts — only the final result is posted/edited.

Auto-watch (no explicit prompt) is DISABLED.

Run manually:  python3 whisper_watcher.py
Run once:      python3 whisper_watcher.py --once
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import re
import select as _select
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from whisper_cli.dedupe import file_hash, has_override, is_duplicate, mark_processed as dedupe_mark
from whisper_cli.downloader import is_supported_url

# ── Config ────────────────────────────────────────────────────────────────

WHISPER_CHANNEL = "1490901110414905580"
STATE_DIR = Path(__file__).parent / ".whisper_state"
STATE_FILE = STATE_DIR / "watcher_state.json"
LOCK_FILE = Path(__file__).parent / ".whisper_watcher.lock"
LOCKS_DIR = STATE_DIR / "locks"
POLL_INTERVAL = 30  # seconds
DEBOUNCE_SECONDS = 120

WATCHDOG_DEFAULT = 1800      # total wall-clock limit per task
DOWNLOAD_PHASE_MAX = 600     # cap on download + caption-fetch phase
PROGRESS_HANG_TIMEOUT = 300  # kill yt-dlp if no stdout progress for this long
TIMEOUT_MIN = 300
TIMEOUT_MAX = 3600

# QUIET_MODE: when this file exists, suppress "Working…" and intermediate posts.
# Only the final result message is posted.
# To re-enable: delete .whisper_quiet_mode or run with --disable-quiet-mode.
QUIET_MODE_FILE = Path(__file__).parent / ".whisper_quiet_mode"

GEORGE_USER_ID = "732025456617979925"

TRIGGER_KEYWORDS = ("summarize", "tl;dr", "tldr", "transcribe", "process")

MAX_RETRIES = 3


def is_quiet_mode() -> bool:
    return QUIET_MODE_FILE.exists()


# ── State ─────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"processed_ids": [], "last_message_id": None, "debounce": {}}


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── Per-source lock ───────────────────────────────────────────────────────

def _source_lock_path(source_id: str) -> Path:
    key = hashlib.sha256(source_id.encode()).hexdigest()[:16]
    LOCKS_DIR.mkdir(parents=True, exist_ok=True)
    return LOCKS_DIR / f"{key}.lock"


def acquire_source_lock(source_id: str):
    """Return an open, exclusively-locked file handle, or None if already locked."""
    lock_path = _source_lock_path(source_id)
    fp = open(lock_path, "w")
    try:
        fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fp
    except OSError:
        fp.close()
        return None


def release_source_lock(fp) -> None:
    if fp:
        try:
            fcntl.flock(fp, fcntl.LOCK_UN)
        except Exception:
            pass
        fp.close()


# ── Debounce ──────────────────────────────────────────────────────────────

def is_debounced(source_id: str, state: dict) -> bool:
    """Return True if the same source was triggered within DEBOUNCE_SECONDS."""
    entry = state.get("debounce", {}).get(source_id)
    if not entry:
        return False
    try:
        then = datetime.fromisoformat(entry["triggered_at"])
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        elapsed = (datetime.now(tz=timezone.utc) - then).total_seconds()
        return elapsed < DEBOUNCE_SECONDS
    except Exception:
        return False


def record_debounce(source_id: str, discord_msg_id: str | None, state: dict) -> None:
    if "debounce" not in state:
        state["debounce"] = {}
    state["debounce"][source_id] = {
        "triggered_at": datetime.now(tz=timezone.utc).isoformat(),
        "discord_msg_id": discord_msg_id,
    }
    # Prune old entries (keep last 100)
    if len(state["debounce"]) > 100:
        oldest = sorted(
            state["debounce"].items(),
            key=lambda kv: kv[1].get("triggered_at", ""),
        )
        for k, _ in oldest[:-100]:
            del state["debounce"][k]


# ── Discord messaging ─────────────────────────────────────────────────────

def post_message(text: str) -> str | None:
    """Post to #whisper. Returns Discord message_id, or None on failure."""
    result = subprocess.run(
        ["clawdbot", "message", "send", "--channel", "discord",
         "--target", WHISPER_CHANNEL, "--message", text, "--json"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[whisper-watcher] post failed: {result.stderr.strip()}")
        return None
    try:
        data = json.loads(result.stdout)
        # clawdbot returns payload.message.id or payload.id
        payload = data.get("payload", {})
        msg = payload.get("message", payload)
        return str(msg.get("id") or msg.get("messageId") or "")
    except Exception as e:
        print(f"[whisper-watcher] could not parse message id: {e}")
        return None


def edit_message(message_id: str, text: str) -> None:
    """Edit an existing Discord message. Falls back to a new post if id missing."""
    if not message_id:
        post_message(text)
        return
    result = subprocess.run(
        ["clawdbot", "message", "edit", "--channel", "discord",
         "--target", WHISPER_CHANNEL, "--message-id", message_id,
         "--message", text],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[whisper-watcher] edit failed (msg {message_id}): {result.stderr.strip()}")
        # Fall back: post new message
        post_message(text)


def post_or_edit(discord_msg_id: str | None, text: str) -> str | None:
    """Edit existing message if we have an id, otherwise post new."""
    if discord_msg_id:
        edit_message(discord_msg_id, text)
        return discord_msg_id
    return post_message(text)


# ── Download ──────────────────────────────────────────────────────────────

def download_attachment(url: str, dest_dir: Path) -> Path:
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
    """Download URL to dest_dir. Kills yt-dlp if no stdout progress for PROGRESS_HANG_TIMEOUT seconds."""
    template = str(dest_dir / "%(title).60s.%(ext)s")
    proc = subprocess.Popen(
        ["yt-dlp", "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
         "--merge-output-format", "mp4", "--output", template, "--no-playlist",
         "--print", "after_move:filepath", "--newline", url],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    last_progress = time.time()
    output_lines: list[str] = []
    while proc.poll() is None:
        ready, _, _ = _select.select([proc.stdout], [], [], 5.0)
        if ready:
            line = proc.stdout.readline()
            if line:
                output_lines.append(line)
                last_progress = time.time()
                print(f"[whisper-watcher] yt-dlp: {line.rstrip()}", flush=True)
        if time.time() - last_progress > PROGRESS_HANG_TIMEOUT:
            proc.kill()
            proc.wait()
            raise RuntimeError(f"Download stalled: no progress for {PROGRESS_HANG_TIMEOUT}s")
    remaining = proc.stdout.read()
    if remaining:
        output_lines.extend(remaining.splitlines(keepends=True))
    if proc.returncode != 0:
        tail = "".join(output_lines[-5:]).strip()
        raise RuntimeError(f"yt-dlp failed: {tail}")
    filepath = "".join(output_lines).strip().splitlines()[-1] if output_lines else ""
    path = Path(filepath)
    if not path.exists():
        files = sorted(dest_dir.glob("*.mp4"), key=lambda f: f.stat().st_mtime)
        if not files:
            raise RuntimeError("yt-dlp succeeded but no mp4 found")
        path = files[-1]
    return path


def try_fetch_captions(url: str, dest_dir: Path) -> str | None:
    """Try to fetch auto/manual captions for a URL. Returns plain text or None."""
    template = str(dest_dir / "%(title).60s")
    result = subprocess.run(
        ["yt-dlp", "--write-auto-subs", "--write-subs",
         "--sub-langs", "en,en-US,en-GB,en-orig",
         "--sub-format", "vtt/best",
         "--skip-download", "--output", template, "--no-playlist", url],
        capture_output=True, text=True, timeout=DOWNLOAD_PHASE_MAX,
    )
    for pattern in ("*.vtt", "*.srt", "*.ttml"):
        files = list(dest_dir.glob(pattern))
        if files:
            raw = files[0].read_text(errors="replace")
            lines: list[str] = []
            seen: set[str] = set()
            for line in raw.splitlines():
                line = line.strip()
                if not line or line.startswith("WEBVTT") or "-->" in line:
                    continue
                clean = re.sub(r"<[^>]+>", "", line).strip()
                if clean and clean not in seen:
                    seen.add(clean)
                    lines.append(clean)
            if lines:
                return " ".join(lines)
    return None


# ── Snippety update ───────────────────────────────────────────────────────

def _update_snippety(video_path: Path, summary: str) -> None:
    """Append/update Snippety CSV with this video's summary, if SNIPPETY_CSV_PATH is set."""
    try:
        from whisper_cli.config import load_config
        from whisper_cli.snippety import export_snippets_csv
        cfg = load_config()
        if not cfg.snippety_csv_path:
            return
        export_snippets_csv({video_path.stem: summary}, cfg.snippety_csv_path)
        print(f"[whisper-watcher] Snippety updated: {cfg.snippety_csv_path}")
    except Exception as e:
        print(f"[whisper-watcher] Snippety update failed (non-fatal): {e}")


# ── Transcribe + Summarize ────────────────────────────────────────────────

def transcribe_and_summarize(
    video_path: Path | None,
    discord_msg_id: str | None,
    *,
    caption_text: str | None = None,
) -> str:
    """Transcribe + summarize with retries. If caption_text is provided, ASR is skipped.
    Edits discord_msg_id on each retry. Returns formatted summary. Raises on final failure."""
    from whisper_cli.config import load_config
    from whisper_cli.transcriber import transcribe
    from whisper_cli.summarizer import summarize

    cfg = load_config()
    last_exc: Exception | None = None
    label = video_path.name if video_path else "media"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if caption_text:
                transcript = caption_text
                print(f"[whisper-watcher] Using captions ({len(transcript)} chars) — ASR skipped")
            else:
                print(f"[whisper-watcher] Transcribing {label} (attempt {attempt})...")
                transcript = transcribe(video_path, cfg.whisper_model)

            print(f"[whisper-watcher] Summarizing ({len(transcript)} chars)...")
            summary = summarize(transcript, label, cfg.openai_api_key)
            return summary

        except Exception as e:
            last_exc = e
            print(f"[whisper-watcher] attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                retry_text = f"⏳ Processing… (retry {attempt}/{MAX_RETRIES - 1})"
                if not is_quiet_mode():
                    post_or_edit(discord_msg_id, retry_text)
                time.sleep(2 ** attempt)

    raise last_exc  # type: ignore[misc]


# ── URL Detection ─────────────────────────────────────────────────────────

def extract_urls(content: str) -> list[str]:
    urls = []
    for word in content.split():
        clean = word.strip("<>")
        if clean.startswith("http") and is_supported_url(clean):
            urls.append(clean)
    return urls


# ── Gate checks ───────────────────────────────────────────────────────────

def is_from_george(msg: dict) -> bool:
    author = msg.get("author") or {}
    return str(author.get("id", "")) == GEORGE_USER_ID


def has_trigger(content: str) -> bool:
    lower = content.lower()
    if any(kw in lower for kw in TRIGGER_KEYWORDS):
        return True
    try:
        data = json.loads(content.strip())
        return isinstance(data, dict) and data.get("action") in ("process", "transcribe", "summarize")
    except Exception:
        return False


# ── Dispatch parsing ─────────────────────────────────────────────────────

def parse_dispatch_timeout(content: str) -> int:
    """Extract timeoutSec from JSON dispatch, clamped to [TIMEOUT_MIN, TIMEOUT_MAX]."""
    try:
        data = json.loads(content.strip())
        if isinstance(data, dict) and "timeoutSec" in data:
            return max(TIMEOUT_MIN, min(TIMEOUT_MAX, int(data["timeoutSec"])))
    except Exception:
        pass
    return WATCHDOG_DEFAULT


# ── Core source processing ────────────────────────────────────────────────

def process_source(
    source_id: str,
    source_label: str,
    source_type: str,
    get_video: callable,
    force: bool,
    state: dict,
    tmp: Path,
    timeout_sec: int = WATCHDOG_DEFAULT,
) -> bool:
    """Process a single source (attachment or URL). Returns True if processed.

    Watchdog: kills the task after timeout_sec total wall-clock seconds.
    Phases: caption-fetch/download (hard cap DOWNLOAD_PHASE_MAX via subprocess timeout
    and hang-guard), ASR (covered by remaining outer watchdog time).
    Lock is always released in finally — no stale locks on watchdog expiry.
    """

    # Debounce: same source triggered within 120s → skip
    if not force and is_debounced(source_id, state):
        print(f"[whisper-watcher] Debounce: {source_label} triggered within {DEBOUNCE_SECONDS}s — skipping")
        return False

    # Per-source lock: skip if another process is already handling this source
    lock_fp = acquire_source_lock(source_id)
    if lock_fp is None:
        print(f"[whisper-watcher] Lock busy: {source_label} — job already in flight")
        return False

    discord_msg_id: str | None = None

    def _watchdog_handler(signum, frame):
        raise TimeoutError(f"Watchdog: task killed after {timeout_sec}s")

    old_handler = signal.signal(signal.SIGALRM, _watchdog_handler)
    signal.alarm(timeout_sec)
    print(f"[whisper-watcher] Watchdog set: {timeout_sec}s")

    try:
        # Last-item dedupe (24h window)
        if not force and is_duplicate(source_id, source_type):
            print(f"[whisper-watcher] Dedupe: {source_label} already processed within 24h")
            short = source_label[:60] + "…" if len(source_label) > 60 else source_label
            post_message(
                f"⚠️ `{short}` matches last processed item (within 24h). "
                "Reply: process anyway"
            )
            record_debounce(source_id, None, state)
            return False

        # Post "Working…" placeholder (skipped in QUIET_MODE)
        if not is_quiet_mode():
            short = source_label[:60] + "…" if len(source_label) > 60 else source_label
            discord_msg_id = post_message(f"⏳ Working on `{short}`…")

        # Record debounce entry with discord_msg_id
        record_debounce(source_id, discord_msg_id, state)
        save_state(state)

        # Caption-first for URL sources (skip ASR if captions available)
        caption_text: str | None = None
        video_path: Path | None = None
        content_hash = None

        if source_type == "url":
            print("[whisper-watcher] Phase 1: trying captions…")
            try:
                caption_text = try_fetch_captions(source_id, tmp)
                if caption_text:
                    print(f"[whisper-watcher] Captions found ({len(caption_text)} chars) — ASR skipped")
                else:
                    print("[whisper-watcher] No captions — falling back to ASR")
            except Exception as e:
                print(f"[whisper-watcher] Caption fetch failed (non-fatal): {e}")

        if not caption_text:
            print("[whisper-watcher] Phase 1: downloading media…")
            video_path = get_video(tmp)
            content_hash = file_hash(video_path) if source_type == "file" else None

        # Transcribe + summarize (retries reuse discord_msg_id)
        summary = transcribe_and_summarize(video_path, discord_msg_id, caption_text=caption_text)

        # Auto-update Snippety CSV
        if video_path:
            _update_snippety(video_path, summary)

        # Build final message
        short = source_label[:60] + "…" if len(source_label) > 60 else source_label
        final_text = f"✅ **{short}**\n{summary}"

        # Edit or post final result
        discord_msg_id = post_or_edit(discord_msg_id, final_text)

        dedupe_mark(source_id, source_type, content_hash)
        return True

    except TimeoutError as e:
        print(f"[whisper-watcher] {e}")
        err_text = f"⏱️ Timed out: `{source_label[:60]}`\nKilled after {timeout_sec}s."
        post_or_edit(discord_msg_id, err_text)
        return False

    except Exception as e:
        print(f"[whisper-watcher] Failed to process {source_label}: {e}")
        err_text = f"⚠️ Failed: `{source_label[:60]}`\n{e}"
        post_or_edit(discord_msg_id, err_text)
        return False

    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        release_source_lock(lock_fp)


# ── Main loop ─────────────────────────────────────────────────────────────

def process_message(msg: dict, state: dict) -> bool:
    """Process a single Discord message. Returns True if any source was processed."""
    msg_id = msg["id"]
    content = msg.get("content", "")
    attachments = msg.get("attachments", [])
    force = has_override(content)
    timeout_sec = parse_dispatch_timeout(content)

    any_work = False

    with tempfile.TemporaryDirectory(prefix="whisper_dl_") as tmpdir:
        tmp = Path(tmpdir)

        # Process video attachments
        for att in attachments:
            if not att.get("content_type", "").startswith("video/"):
                continue
            source_id = att["filename"]
            att_url = att["url"]
            print(f"[whisper-watcher] Attachment: {source_id}")

            did_work = process_source(
                source_id=source_id,
                source_label=source_id,
                source_type="file",
                get_video=lambda t, u=att_url: download_attachment(u, t),
                force=force,
                state=state,
                tmp=tmp,
                timeout_sec=timeout_sec,
            )
            if did_work:
                any_work = True

        # Process YouTube/TikTok URLs
        for url in extract_urls(content):
            source_id = url.rstrip("/")
            print(f"[whisper-watcher] URL: {source_id}")

            did_work = process_source(
                source_id=source_id,
                source_label=source_id,
                source_type="url",
                get_video=lambda t, u=url: download_url(u, t),
                force=force,
                state=state,
                tmp=tmp,
                timeout_sec=timeout_sec,
            )
            if did_work:
                any_work = True

    # Mark message as seen after attempting work
    state["processed_ids"].append(msg_id)
    state["processed_ids"] = state["processed_ids"][-2000:]
    save_state(state)

    return any_work


def poll_once(state: dict) -> None:
    cmd = ["clawdbot", "message", "read", "--channel", "discord",
           "--target", WHISPER_CHANNEL, "--json", "--limit", "20"]
    after = state.get("last_message_id")
    if after:
        cmd += ["--after", after]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[whisper-watcher] clawdbot read failed: {result.stderr.strip()}")
        return

    try:
        data = json.loads(result.stdout)
        messages = data["payload"]["messages"]
    except Exception as e:
        print(f"[whisper-watcher] parse error: {e}")
        return

    if not messages:
        return

    latest_id = max(m["id"] for m in messages)
    state["last_message_id"] = latest_id

    processed_ids = set(state.get("processed_ids", []))
    for msg in sorted(messages, key=lambda m: m["id"]):
        if msg["id"] in processed_ids:
            continue

        if not is_from_george(msg):
            continue

        msg_content = msg.get("content", "")
        has_video = any(a.get("content_type", "").startswith("video/") for a in msg.get("attachments", []))
        has_url = bool(extract_urls(msg_content))
        if not (has_video or has_url):
            continue

        if not has_trigger(msg_content):
            print(f"[whisper-watcher] Skipping msg {msg['id']} — no explicit trigger")
            continue

        process_message(msg, state)

    save_state(state)


def main():
    parser = argparse.ArgumentParser(description="whisper-watcher: Discord #whisper listener")
    parser.add_argument("--once", action="store_true", help="Poll once and exit")
    parser.add_argument("--disable-quiet-mode", action="store_true",
                        help="Remove .whisper_quiet_mode sentinel and resume normal posting")
    args = parser.parse_args()

    if args.disable_quiet_mode:
        if QUIET_MODE_FILE.exists():
            QUIET_MODE_FILE.unlink()
            print("[whisper-watcher] QUIET_MODE disabled — resuming normal operation.")
        else:
            print("[whisper-watcher] QUIET_MODE was not active.")

    # Process-level singleton lock
    lock_fp = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("[whisper-watcher] Another instance is already running. Exiting.")
        sys.exit(1)

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()

    print(f"[whisper-watcher] Starting. Channel: #whisper ({WHISPER_CHANNEL})")
    print(f"[whisper-watcher] George-only, explicit-trigger mode. Debounce: {DEBOUNCE_SECONDS}s.")
    if is_quiet_mode():
        print("[whisper-watcher] QUIET_MODE active — Working… placeholders suppressed.")

    if args.once:
        poll_once(state)
        return

    try:
        while True:
            poll_once(state)
            print(f"[whisper-watcher] Sleeping {POLL_INTERVAL}s…")
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print("\n[whisper-watcher] Stopped.")
    finally:
        fcntl.flock(lock_fp, fcntl.LOCK_UN)
        lock_fp.close()


if __name__ == "__main__":
    main()
