# Project Status

> Current direction and state of the project.

## Current Milestone
Feature-complete — all planned backlog items shipped.

## Status
archived — manual invocation only. Not dispatched by Trax. Run: `cd /Users/g2/Trax/whisper-cli && .venv/bin/python -m whisper_cli.cli run <folder>`

## Last Updated
2026-04-09

## Summary
Full pipeline live: drop a video/URL in #whisper → transcribed via OpenAI Whisper API → summarized as language + duration + 3–6 bullet TL;DR → posted back to #whisper → Snippety CSV updated. Supports YouTube, TikTok, Twitter/X, Instagram, Reddit, Vimeo, Twitch. Long videos use chunked summarization. `clip` command cuts video segments from timestamped notes via ffmpeg. 60/60 tests passing (16 skipped — need real API key).

## Blockers
None
