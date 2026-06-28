# Project Status

> Current direction and state of the project.

## Current Milestone
Feature-complete — all planned backlog items shipped.

## Status
active — monitored by Trax maintenance + health checks. Explicit dispatches and manual runs remain supported; it is not in the hard-coded main `trax-run.sh` rotation. Run manually with `cd /Users/g2/Trax/whisper-cli && .venv/bin/python -m whisper_cli.cli run <folder>`

## Last Updated
2026-04-21

## Summary
Full pipeline live: drop a video/URL in #whisper → transcribed via OpenAI Whisper API → summarized with quote-heavy output → posted back to #whisper → Snippety CSV updated. Supports YouTube, TikTok, Twitter/X, Instagram, Reddit, Vimeo, Twitch. Long videos use chunked summarization. `clip` command cuts video segments from timestamped notes via ffmpeg. 60/60 tests passing (16 skipped — need real API key).

## Blockers
None
