# CLAUDE.md

> Coding instructions for Coder in this repository.

## Project Overview
Video transcription bot. Drop a video file, YouTube link, or TikTok link in the #whisper Discord channel — it gets downloaded, transcribed via Whisper, then summarized as a TL;DR (language, duration, 3–6 bullets). Snippety CSV updated for persistent snippet access. Future: video editing from notes.

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
whisper_cli/
├── cli.py          # typer app — run, watch, list, reset commands
├── config.py       # env/config loading, ffmpeg check
├── scanner.py      # video file discovery (mp4, mov, mkv, avi, webm)
├── transcriber.py  # Whisper model loading + transcription
├── summarizer.py   # gpt-4o-mini summarization
├── snippety.py     # CSV merge for Snippety export
└── state.py        # JSON state tracking (idempotent reprocessing)
```

Discord input flow (to be built): Discord message with attachment/URL → download → transcribe → summarize → post result to #whisper → update Snippety CSV.

## Rules
- Commit after each meaningful change
- Run tests before marking a task complete
- Summarization outputs: language, duration, and a 3–6 bullet TL;DR. No quotes section.
- Update `DECISIONS.md` when making architectural choices
- Update `NEXT_STEPS.md` when completing or adding tasks
