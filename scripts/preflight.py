#!/usr/bin/env python3
"""
preflight.py — Sprint preflight healthcheck for whisper-cli.

Checks all hard dependencies before a sprint iteration starts.
Exits 0 on success, 1 on failure (prints a single summarized error).

Usage:  python3 scripts/preflight.py [--verbose]
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / ".whisper_state"
WHISPER_CHANNEL = "1490901110414905580"


def check(label: str, ok: bool, detail: str = "", verbose: bool = False) -> bool:
    if verbose:
        status = "✅" if ok else "❌"
        msg = f"  {status} {label}"
        if detail:
            msg += f": {detail}"
        print(msg)
    return ok


def run_preflight(verbose: bool = False) -> list[str]:
    failures: list[str] = []

    # ffmpeg
    ffmpeg_path = shutil.which("ffmpeg")
    if not check("ffmpeg", bool(ffmpeg_path), ffmpeg_path or "not found", verbose):
        failures.append("ffmpeg not found (brew install ffmpeg)")

    # yt-dlp (system PATH or venv)
    venv_ytdlp = ROOT / ".venv/bin/yt-dlp"
    ytdlp_path = shutil.which("yt-dlp") or (str(venv_ytdlp) if venv_ytdlp.exists() else None)
    if not check("yt-dlp", bool(ytdlp_path), ytdlp_path or "not found", verbose):
        failures.append("yt-dlp not found (pip install yt-dlp)")

    # OPENAI_API_KEY
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    api_key = os.getenv("OPENAI_API_KEY", "")
    key_ok = bool(api_key) and api_key != "your-key-here"
    if not check("OPENAI_API_KEY", key_ok, "set" if key_ok else "missing or placeholder", verbose):
        failures.append("OPENAI_API_KEY not set")

    # State dir writable
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        test_file = STATE_DIR / ".preflight_write_test"
        test_file.write_text("ok")
        test_file.unlink()
        check("state dir writable", True, str(STATE_DIR), verbose)
    except Exception as e:
        check("state dir writable", False, str(e), verbose)
        failures.append(f"State dir not writable: {STATE_DIR}")

    # clawdbot available
    clawdbot_path = shutil.which("clawdbot")
    if not check("clawdbot", bool(clawdbot_path), clawdbot_path or "not found", verbose):
        failures.append("clawdbot not found — Discord posting will fail")

    # Discord channel reachable (only if clawdbot is available)
    if clawdbot_path:
        try:
            result = subprocess.run(
                ["clawdbot", "message", "send", "--channel", "discord",
                 "--target", WHISPER_CHANNEL, "--message", "🔍 preflight check"],
                capture_output=True, text=True, timeout=60,
            )
            discord_ok = result.returncode == 0
            check("Discord #whisper postable", discord_ok,
                  "ok" if discord_ok else result.stderr.strip()[:80], verbose)
            if not discord_ok:
                # Non-fatal: Discord issues shouldn't block code sprints
                print(f"  ⚠️  Discord post failed (non-fatal): {result.stderr.strip()[:80]}")
        except subprocess.TimeoutExpired:
            # Non-fatal: clawdbot can be slow in subprocess context
            check("Discord #whisper postable", False, "timeout (non-fatal)", verbose)
        except Exception as e:
            check("Discord #whisper postable", False, str(e), verbose)
            failures.append(f"Discord check error: {e}")

    return failures


def main() -> None:
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    if verbose:
        print("=== whisper-cli preflight ===")

    failures = run_preflight(verbose=verbose)

    if failures:
        print(f"PREFLIGHT FAILED ({len(failures)} issue(s)):")
        for f in failures:
            print(f"  • {f}")
        sys.exit(1)
    else:
        if verbose:
            print("All checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
