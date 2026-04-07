# Project Status

> Current direction and state of the project.

## Current Milestone
Discord listener live — whisper_watcher.py watches #whisper for video attachments/URLs, transcribes, and posts summaries back.

## Status
active

## Last Updated
2026-04-06

## Summary
Discord listener is live (`whisper_watcher.py`). Drop a video in #whisper or paste a YouTube/TikTok URL — watcher polls every 30s, transcribes via Whisper API, summarizes with quote-heavy output, posts back to #whisper. 33/33 tests passing. Transcriber migrated from local model to OpenAI Whisper API (avoids llvmlite install issues).

## Blockers
None
