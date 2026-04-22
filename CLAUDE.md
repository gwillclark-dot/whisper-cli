# CLAUDE.md

> Coding instructions for Coder in this repository.

## Project Overview
Video transcription bot. Drop a video file or URL in the #whisper Discord channel — it gets downloaded, transcribed via Whisper, then summarized with heavy quoting. Snippety CSV updated for persistent snippet access. Supports YouTube, TikTok, Twitter/X, Instagram, Reddit, Vimeo, Twitch. `clip` command cuts video segments from timestamped notes via ffmpeg.

## Tech Stack
- Python 3.13, typer CLI framework
- OpenAI Whisper (local transcription)
- OpenAI gpt-4o-mini (summarization)
- yt-dlp (YouTube/TikTok download)
- ffmpeg (audio/video processing)
- clawdbot (Discord integration)
- Snippety CSV export

## Development Commands
```bash
# Install dependencies
cd /Users/g2/Trax/whisper-cli && .venv/bin/pip install -e ".[dev]"

# Run tests
.venv/bin/pytest tests/test_unit.py -v

# Run full test suite (needs OPENAI_API_KEY)
.venv/bin/pytest tests/ -v

# Run CLI
.venv/bin/python -m whisper_cli.cli run <folder>
```

## Architecture
```
whisper_watcher.py        # Discord poller — polls #whisper every 30s, drives pipeline
whisper_cli/
├── cli.py          # typer app — run, watch, list, reset, clip commands
├── config.py       # env/config loading, ffmpeg check
├── scanner.py      # video file discovery (mp4, mov, mkv, avi, webm)
├── transcriber.py  # Whisper API transcription (whisper-1)
├── summarizer.py   # gpt-4o-mini summarization, chunked for long transcripts
├── downloader.py   # yt-dlp download; netloc-based URL detection for supported domains
├── dedupe.py       # 24h TTL dedupe guard + per-source lock
├── clipper.py      # ffmpeg segment cutter from timestamped notes
├── snippety.py     # CSV merge for Snippety export
└── state.py        # JSON state tracking (idempotent reprocessing)
```

Discord flow: message with attachment/URL → `whisper_watcher.py` → download → transcribe → summarize → post result to #whisper → update Snippety CSV.

## Rules
- Commit after each meaningful change
- Run tests before marking a task complete
- Summarization must emphasize **direct quoting** — pull actual phrases from transcripts, not just paraphrase
- Update `DECISIONS.md` when making architectural choices
- Update `NEXT_STEPS.md` when completing or adding tasks
