"""Layer 1: Node-level unit tests — deterministic, no external calls."""

import csv
import json
import tempfile
from pathlib import Path

import pytest

from whisper_cli.scanner import VideoFile, scan_folder
from whisper_cli.state import (
    ProcessedEntry,
    State,
    get_unprocessed,
    load_state,
    mark_processed,
    output_base,
    save_state,
    state_path,
)
from whisper_cli.snippety import VIDEO_KEYWORD_PREFIX, export_snippets_csv


# ── Scanner ──────────────────────────────────────────────────────────────


class TestScanner:
    def test_finds_video_files(self, tmp_path):
        (tmp_path / "clip.mp4").write_text("fake")
        (tmp_path / "clip.mov").write_text("fake")
        (tmp_path / "notes.txt").write_text("not a video")
        result = scan_folder(tmp_path)
        assert len(result) == 2
        names = {v.path.name for v in result}
        assert names == {"clip.mp4", "clip.mov"}

    def test_ignores_subdirectories(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.mp4").write_text("fake")
        (tmp_path / "top.mp4").write_text("fake")
        result = scan_folder(tmp_path)
        assert len(result) == 1
        assert result[0].path.name == "top.mp4"

    def test_empty_folder(self, tmp_path):
        assert scan_folder(tmp_path) == []

    def test_case_insensitive_extensions(self, tmp_path):
        (tmp_path / "clip.MP4").write_text("fake")
        (tmp_path / "clip.MoV").write_text("fake")
        result = scan_folder(tmp_path)
        assert len(result) == 2

    def test_all_supported_extensions(self, tmp_path):
        for ext in (".mp4", ".mov", ".mkv", ".avi", ".webm"):
            (tmp_path / f"clip{ext}").write_text("fake")
        result = scan_folder(tmp_path)
        assert len(result) == 5

    def test_returns_correct_metadata(self, tmp_path):
        f = tmp_path / "clip.mp4"
        f.write_text("some content")
        result = scan_folder(tmp_path)
        assert result[0].size_bytes == f.stat().st_size
        assert result[0].mtime == f.stat().st_mtime

    def test_sorted_by_mtime(self, tmp_path):
        import time

        (tmp_path / "old.mp4").write_text("old")
        time.sleep(0.05)
        (tmp_path / "new.mp4").write_text("new")
        result = scan_folder(tmp_path)
        assert result[0].path.name == "old.mp4"
        assert result[1].path.name == "new.mp4"


# ── State ────────────────────────────────────────────────────────────────


class TestState:
    def test_roundtrip_save_load(self, tmp_path):
        sp = tmp_path / "state.json"
        state = State()
        video = VideoFile(path=tmp_path / "clip.mp4", mtime=1000.0, size_bytes=500)
        mark_processed(state, video, "ok", transcript_chars=100, summary_chars=50)
        save_state(state, sp)
        loaded = load_state(sp)
        assert str(video.path) in loaded.processed
        entry = loaded.processed[str(video.path)]
        assert entry.status == "ok"
        assert entry.transcript_chars == 100
        assert entry.summary_chars == 50

    def test_load_missing_file_returns_empty(self, tmp_path):
        state = load_state(tmp_path / "nonexistent.json")
        assert state.processed == {}

    def test_load_corrupt_file_backs_up(self, tmp_path):
        sp = tmp_path / "state.json"
        sp.write_text("{invalid json!!")
        state = load_state(sp)
        assert state.processed == {}
        assert (tmp_path / "state.json.bak").exists()

    def test_get_unprocessed_filters_done(self, tmp_path):
        state = State()
        v1 = VideoFile(path=tmp_path / "done.mp4", mtime=1000.0, size_bytes=500)
        v2 = VideoFile(path=tmp_path / "new.mp4", mtime=2000.0, size_bytes=600)
        mark_processed(state, v1, "ok")
        pending = get_unprocessed([v1, v2], state)
        assert len(pending) == 1
        assert pending[0].path.name == "new.mp4"

    def test_get_unprocessed_retries_errors(self, tmp_path):
        state = State()
        v = VideoFile(path=tmp_path / "fail.mp4", mtime=1000.0, size_bytes=500)
        mark_processed(state, v, "error_transcribe", error="boom")
        pending = get_unprocessed([v], state)
        assert len(pending) == 1

    def test_get_unprocessed_reprocesses_changed_mtime(self, tmp_path):
        state = State()
        v_old = VideoFile(path=tmp_path / "clip.mp4", mtime=1000.0, size_bytes=500)
        mark_processed(state, v_old, "ok")
        v_new = VideoFile(path=tmp_path / "clip.mp4", mtime=2000.0, size_bytes=500)
        pending = get_unprocessed([v_new], state)
        assert len(pending) == 1

    def test_output_base_default(self, tmp_path):
        base = output_base(tmp_path)
        assert base == tmp_path / "vidsum-output"
        assert base.exists()

    def test_output_base_override(self, tmp_path):
        custom = tmp_path / "custom-out"
        base = output_base(tmp_path, custom)
        assert base == custom
        assert base.exists()

    def test_state_path_inside_base(self, tmp_path):
        base = output_base(tmp_path)
        sp = state_path(base)
        assert sp == base / "state.json"


# ── Snippety CSV ─────────────────────────────────────────────────────────


class TestSnippetyCSV:
    def _read_csv(self, path: Path) -> list[dict]:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))

    def test_creates_csv_from_summaries(self, tmp_path):
        csv_path = tmp_path / "snippets.csv"
        summaries = {"meeting_recap": "Discussed Q2 goals", "standup": "All green"}
        export_snippets_csv(summaries, csv_path)
        rows = self._read_csv(csv_path)
        assert len(rows) == 2
        keywords = {r["keyword"] for r in rows}
        assert keywords == {"vid-meeting_recap", "vid-standup"}

    def test_preserves_existing_non_video_snippets(self, tmp_path):
        csv_path = tmp_path / "snippets.csv"
        # Write existing user snippets
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["keyword", "title", "content"])
            writer.writeheader()
            writer.writerow({"keyword": "addr", "title": "Home Address", "content": "123 Main St"})
            writer.writerow({"keyword": "sig", "title": "Email Sig", "content": "Best, George"})

        # Merge video summaries
        export_snippets_csv({"demo": "A demo video summary"}, csv_path)
        rows = self._read_csv(csv_path)
        assert len(rows) == 3
        keywords = {r["keyword"] for r in rows}
        assert "addr" in keywords
        assert "sig" in keywords
        assert "vid-demo" in keywords

    def test_updates_existing_video_snippets(self, tmp_path):
        csv_path = tmp_path / "snippets.csv"
        # First run
        export_snippets_csv({"clip1": "Old summary"}, csv_path)
        # Second run with updated summary
        export_snippets_csv({"clip1": "New summary", "clip2": "Another"}, csv_path)
        rows = self._read_csv(csv_path)
        assert len(rows) == 2
        clip1 = next(r for r in rows if r["keyword"] == "vid-clip1")
        assert clip1["content"] == "New summary"

    def test_handles_missing_csv_file(self, tmp_path):
        csv_path = tmp_path / "nonexistent.csv"
        export_snippets_csv({"clip": "Summary"}, csv_path)
        rows = self._read_csv(csv_path)
        assert len(rows) == 1

    def test_preserves_multiline_content(self, tmp_path):
        csv_path = tmp_path / "snippets.csv"
        multi = "- Point one\n- Point two\n- Point three"
        export_snippets_csv({"clip": multi}, csv_path)
        rows = self._read_csv(csv_path)
        assert rows[0]["content"] == multi

    def test_empty_summaries_no_crash(self, tmp_path):
        csv_path = tmp_path / "snippets.csv"
        export_snippets_csv({}, csv_path)
        rows = self._read_csv(csv_path)
        assert len(rows) == 0

    def test_special_chars_in_summary(self, tmp_path):
        csv_path = tmp_path / "snippets.csv"
        content = 'He said "hello" & she said <goodbye>, cost was $5.00'
        export_snippets_csv({"clip": content}, csv_path)
        rows = self._read_csv(csv_path)
        assert rows[0]["content"] == content


# ── Summarizer chunking ───────────────────────────────────────────────────


class TestSummarizerChunking:
    def test_split_chunks_short_text(self):
        from whisper_cli.summarizer import _split_chunks
        text = "Hello world. " * 10
        chunks = _split_chunks(text, chunk_size=1000, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_chunks_long_text_produces_multiple(self):
        from whisper_cli.summarizer import _split_chunks
        text = "word " * 10_000  # ~50k chars
        chunks = _split_chunks(text, chunk_size=20_000, overlap=500)
        assert len(chunks) >= 2

    def test_split_chunks_coverage(self):
        from whisper_cli.summarizer import _split_chunks
        text = "abcde " * 5_000  # 30k chars
        chunks = _split_chunks(text, chunk_size=10_000, overlap=200)
        # Reconstruct without overlap and verify all content covered
        assert chunks[0][:100] in text
        assert chunks[-1][-100:] in text
        # Last chunk ends at end of text
        assert text.endswith(chunks[-1])

    def test_split_chunks_overlap_carries_context(self):
        from whisper_cli.summarizer import _split_chunks
        text = "x" * 100
        chunks = _split_chunks(text, chunk_size=60, overlap=30)
        # With overlap, chunks share content — second chunk starts at 60-30=30
        assert chunks[1][:10] == text[30:40]
        # Last chunk ends exactly at end of text
        assert text.endswith(chunks[-1])

    def test_summarize_routes_short_to_direct(self, monkeypatch):
        """Short transcripts should make exactly one API call."""
        from whisper_cli.summarizer import summarize, MAX_DIRECT_CHARS
        calls = []

        def fake_chat(client, system, user, max_tokens=800):
            calls.append(("direct", user))
            return "short summary"

        monkeypatch.setattr("whisper_cli.summarizer._chat", fake_chat)
        result = summarize("short text", "clip.mp4", "fake-key")
        assert result == "short summary"
        assert len(calls) == 1

    def test_summarize_routes_long_to_chunked(self, monkeypatch):
        """Long transcripts should make N+1 API calls (N chunks + meta-summary)."""
        from whisper_cli.summarizer import summarize, CHUNK_SIZE
        calls = []

        def fake_chat(client, system, user, max_tokens=800):
            calls.append(user)
            return "chunk extract"

        monkeypatch.setattr("whisper_cli.summarizer._chat", fake_chat)
        # Create a transcript longer than MAX_DIRECT_CHARS
        long_text = "word " * 30_000  # ~150k chars
        result = summarize(long_text, "long.mp4", "fake-key")
        # Should have called once per chunk + one final meta call
        assert len(calls) >= 3  # at least 2 chunk extracts + 1 meta


# ── Clipper ───────────────────────────────────────────────────────────────


class TestClipperParser:
    def test_parse_mm_ss(self):
        from whisper_cli.clipper import parse_notes
        specs = parse_notes("0:30-1:15 intro\n")
        assert len(specs) == 1
        assert specs[0].start == 30.0
        assert specs[0].end == 75.0
        assert specs[0].label == "intro"

    def test_parse_hh_mm_ss(self):
        from whisper_cli.clipper import parse_notes
        specs = parse_notes("1:02:30-1:03:00 key moment\n")
        assert len(specs) == 1
        assert specs[0].start == 3750.0
        assert specs[0].end == 3780.0
        assert specs[0].label == "key moment"

    def test_parse_raw_seconds(self):
        from whisper_cli.clipper import parse_notes
        specs = parse_notes("90-120 segment\n")
        assert len(specs) == 1
        assert specs[0].start == 90.0
        assert specs[0].end == 120.0

    def test_ignores_comments_and_blanks(self):
        from whisper_cli.clipper import parse_notes
        text = "# header\n\n0:10-0:30 clip1\n# another comment\n1:00-1:30 clip2\n"
        specs = parse_notes(text)
        assert len(specs) == 2

    def test_skips_invalid_lines(self):
        from whisper_cli.clipper import parse_notes
        text = "not a timestamp\n0:10-0:30 valid\njust text\n"
        specs = parse_notes(text)
        assert len(specs) == 1

    def test_skips_reversed_timestamps(self):
        from whisper_cli.clipper import parse_notes
        # end < start should be skipped
        specs = parse_notes("1:00-0:30 backwards\n")
        assert len(specs) == 0

    def test_em_dash_separator(self):
        from whisper_cli.clipper import parse_notes
        specs = parse_notes("0:30–1:00 clip\n")  # em dash
        assert len(specs) == 1

    def test_cut_clip_dry_run(self, tmp_path):
        from whisper_cli.clipper import ClipSpec, cut_clip
        spec = ClipSpec(start=10.0, end=30.0, label="test clip")
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")
        out = cut_clip(video, spec, tmp_path, index=1, dry_run=True)
        assert out.name == "video_clip01_test_clip.mp4"
        # dry_run should not actually write the file
        assert not out.exists()
