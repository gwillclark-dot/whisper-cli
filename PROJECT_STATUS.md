# Project Status

> Current direction and state of the project.

## Current Milestone
Discord listener: watch #whisper channel for video attachments/URLs, auto-process via Whisper, post summary back.

## Status
active

## Last Updated
2026-04-06

## Summary
Core pipeline is complete and tested (23/23 tests passing). Summarizer uses quote-heavy output — leads each bullet with a direct transcript phrase. URL support added (`vidsum url <URL>`) — downloads YouTube/TikTok via yt-dlp then runs the full pipeline. Next: Discord listener integration via clawdbot.

## Blockers
None
