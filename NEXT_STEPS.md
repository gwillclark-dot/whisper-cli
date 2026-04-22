# Next Steps

> Task queue — ordered by priority. Top item = next action.

## Current Sprint

- [x] Rewrite summarizer prompt for quote-heavy output (pull direct phrases from transcript, not just paraphrase)
- [x] Add URL support — yt-dlp download for YouTube and TikTok links before transcription (`vidsum url <URL>`)
- [x] Add Discord listener — `whisper_watcher.py` polls #whisper for attachments/URLs, auto-processes and posts summary back
- [x] Post transcription results back to #whisper via clawdbot

## Backlog

- [x] Video editing from notes — take timestamped notes, cut/clip video segments via ffmpeg
- [x] Chunked summarization for long videos (segment transcript, summarize chunks, then meta-summary)
- [x] Snippety persistent file auto-update on each transcription

## Completed

- [x] Core transcription pipeline (Whisper + gpt-4o-mini summarization)
- [x] State tracking (idempotent reprocessing, error retry)
- [x] CLI commands (run, watch, list, reset)
- [x] Snippety CSV export
- [x] Unit + integration + behavioral test suite (76 collected; 60 pass, 16 skipped — need real API key)
