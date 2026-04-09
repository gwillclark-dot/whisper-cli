# Decisions Log

> Architectural and design decisions. Append-only — never delete entries.

## Format

```
### [YYYY-MM-DD] Decision Title

**Context:** Why this came up.
**Decision:** What was decided.
**Alternatives considered:** What else was on the table.
**Consequences:** What this means going forward.
```

---

<!-- Add decisions below this line -->

### [2026-04-06] OpenAI Whisper API over local model

**Context:** Local Whisper required llvmlite/llvm which caused install failures on some environments.
**Decision:** Use OpenAI Whisper API (`whisper-1`) for transcription instead of the local `whisper` package.
**Alternatives considered:** Keep local model, add llvmlite install workaround.
**Consequences:** Transcription requires OPENAI_API_KEY and incurs API cost; no local GPU needed. Simplifies deps significantly.

### [2026-04-06] Discord watcher uses polling, not websockets

**Context:** Real-time Discord gateway requires bot token + persistent connection; polling is simpler to operate.
**Decision:** `whisper_watcher.py` polls #whisper every 30s using clawdbot REST calls.
**Alternatives considered:** discord.py bot with on_message event.
**Consequences:** Up to 30s latency on new messages. Simpler deployment — no persistent process state or reconnect logic needed.

### [2026-04-07] Per-source lock + 24h dedupe guard

**Context:** Rapid re-posts or repeated Discord messages were triggering duplicate transcriptions.
**Decision:** `dedupe.py` tracks processed sources with 24h TTL; `whisper_watcher.py` holds per-source lock during processing.
**Alternatives considered:** State file check only (no TTL), global lock.
**Consequences:** Same video URL won't be re-processed within 24h. Multiple sources can process concurrently.

### [2026-04-08] Chunked summarization for long transcripts

**Context:** Long videos produce transcripts that exceed gpt-4o-mini context window.
**Decision:** `summarizer.py` splits transcript into ~3000-token chunks, summarizes each, then runs a meta-summary over chunk summaries.
**Alternatives considered:** Truncate transcript, use gpt-4o (larger context), sliding window.
**Consequences:** Long videos are handled correctly at the cost of extra API calls. Quote fidelity slightly reduced in meta-summary.

### [2026-04-08] ffmpeg clip command from timestamped notes

**Context:** Users want to cut video segments based on timestamped notes without manual ffmpeg invocation.
**Decision:** `clipper.py` parses `HH:MM:SS` timestamps from a notes file and calls ffmpeg to cut segments. CLI exposed as `clip` command.
**Alternatives considered:** Interactive editor, GUI tool.
**Consequences:** Fully automatable from notes. Requires ffmpeg in PATH. No re-encoding (stream copy) for speed.

### [2026-04-09] Extended URL support via netloc matching

**Context:** URL detection used substring matching, causing false positives (e.g. `x.com` matched inside `dropbox.com`).
**Decision:** `downloader.py` uses `urlparse` netloc matching for all supported domains. Watcher strips Discord angle-bracket wrapping before URL check.
**Alternatives considered:** Regex per-platform, keep substring match.
**Consequences:** Clean domain matching, no false positives. Adding a new platform = one line in `SUPPORTED_DOMAINS`.
