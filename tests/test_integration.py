"""Layer 2: Pipeline integration tests — mocked external services."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from whisper_cli.scanner import VideoFile
from whisper_cli.state import (
    State,
    get_unprocessed,
    load_state,
    mark_processed,
    output_base,
    save_state,
    state_path,
)

# Mock the whisper module before any import of transcriber
_mock_whisper_module = MagicMock()
sys.modules.setdefault("whisper", _mock_whisper_module)


class TestTranscriptionPipeline:
    """Test the transcription wrapper with mocked whisper model."""

    def setup_method(self):
        # Reset the cached models and the mock before each test
        from whisper_cli.transcriber import _models
        _models.clear()
        _mock_whisper_module.reset_mock()

    def test_transcribe_returns_stripped_text(self):
        from whisper_cli.transcriber import transcribe

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "  Hello world  ", "language": "en", "segments": [{"end": 5.0}]}
        _mock_whisper_module.load_model.return_value = mock_model

        result = transcribe(Path("/fake/video.mp4"), "tiny")
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.duration_secs == 5.0
        mock_model.transcribe.assert_called_once_with("/fake/video.mp4")

    def test_model_caching(self):
        from whisper_cli.transcriber import transcribe

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "text", "language": "en", "segments": []}
        _mock_whisper_module.load_model.return_value = mock_model

        transcribe(Path("/fake/a.mp4"), "tiny")
        transcribe(Path("/fake/b.mp4"), "tiny")
        _mock_whisper_module.load_model.assert_called_once_with("tiny")

    def test_transcribe_empty_result(self):
        from whisper_cli.transcriber import transcribe

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "  ", "language": "unknown", "segments": []}
        _mock_whisper_module.load_model.return_value = mock_model

        result = transcribe(Path("/fake/video.mp4"), "tiny")
        assert result.text == ""


class TestSummarizationPipeline:
    """Test summarizer with mocked OpenAI client."""

    @patch("whisper_cli.summarizer.OpenAI")
    def test_summarize_returns_content(self, mock_openai_cls):
        from whisper_cli.summarizer import summarize

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = "- Key point 1\n- Key point 2"
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        result = summarize("Some transcript text", "video.mp4", "fake-key")
        assert result == "- Key point 1\n- Key point 2"

    @patch("whisper_cli.summarizer.OpenAI")
    def test_summarize_passes_filename(self, mock_openai_cls):
        from whisper_cli.summarizer import summarize

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = "summary"
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        summarize("transcript", "my_meeting.mp4", "fake-key")
        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert "my_meeting.mp4" in user_msg

    @patch("whisper_cli.summarizer.OpenAI")
    def test_summarize_chunks_long_transcript(self, mock_openai_cls):
        from whisper_cli.summarizer import MAX_DIRECT_CHARS, summarize

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = "chunk result"
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        long_text = "word " * 30_000  # ~150k chars, above MAX_DIRECT_CHARS
        assert len(long_text) > MAX_DIRECT_CHARS
        summarize(long_text, "long.mp4", "fake-key")
        # Should make multiple API calls (chunks + meta-summary)
        assert mock_client.chat.completions.create.call_count >= 3

    @patch("whisper_cli.summarizer.OpenAI")
    def test_summarize_retries_on_failure(self, mock_openai_cls):
        from whisper_cli.summarizer import summarize

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "summary"
        success = MagicMock(choices=[mock_choice])

        mock_client.chat.completions.create.side_effect = [
            Exception("timeout"),
            success,
        ]

        result = summarize("transcript", "video.mp4", "fake-key")
        assert result == "summary"
        assert mock_client.chat.completions.create.call_count == 2

    @patch("whisper_cli.summarizer.OpenAI")
    def test_summarize_raises_after_3_failures(self, mock_openai_cls):
        from whisper_cli.summarizer import summarize

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("persistent error")

        with pytest.raises(Exception, match="persistent error"):
            summarize("transcript", "video.mp4", "fake-key")
        assert mock_client.chat.completions.create.call_count == 3


class TestEndToEndPipeline:
    """Test the full scan → filter → (mock transcribe) → (mock summarize) → state flow."""

    def setup_method(self):
        from whisper_cli.transcriber import _models
        _models.clear()
        _mock_whisper_module.reset_mock()

    @patch("whisper_cli.summarizer.OpenAI")
    def test_full_pipeline_writes_outputs(self, mock_openai_cls, tmp_path):
        from whisper_cli.scanner import scan_folder
        from whisper_cli.summarizer import summarize
        from whisper_cli.transcriber import transcribe

        # Setup video folder
        videos_dir = tmp_path / "videos"
        videos_dir.mkdir()
        (videos_dir / "meeting.mp4").write_bytes(b"fake video content")

        # Mock whisper
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "We discussed the Q2 roadmap and agreed on three priorities.",
            "language": "en",
            "segments": [{"end": 120.0}],
        }
        _mock_whisper_module.load_model.return_value = mock_model

        # Mock OpenAI
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = "- Discussed Q2 roadmap\n- Agreed on 3 priorities"
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        # Run pipeline
        base = output_base(videos_dir)
        sp = state_path(base)
        state = load_state(sp)
        videos = scan_folder(videos_dir)
        pending = get_unprocessed(videos, state)

        assert len(pending) == 1

        trans_dir = base / "transcriptions"
        summ_dir = base / "summaries"
        trans_dir.mkdir(parents=True, exist_ok=True)
        summ_dir.mkdir(parents=True, exist_ok=True)

        for v in pending:
            result = transcribe(v.path, "tiny")
            (trans_dir / f"{v.path.stem}.txt").write_text(result.text)

            summary = summarize(result.text, v.path.name, "fake-key")
            (summ_dir / f"{v.path.stem}.txt").write_text(summary)

            mark_processed(state, v, "ok", len(result.text), len(summary))
            save_state(state, sp)

        # Verify outputs
        assert (trans_dir / "meeting.txt").exists()
        assert (summ_dir / "meeting.txt").exists()
        assert "Q2 roadmap" in (trans_dir / "meeting.txt").read_text()

        # Verify state
        reloaded = load_state(sp)
        assert len(reloaded.processed) == 1
        entry = list(reloaded.processed.values())[0]
        assert entry.status == "ok"

        # Second run: no new videos
        videos2 = scan_folder(videos_dir)
        pending2 = get_unprocessed(videos2, reloaded)
        assert len(pending2) == 0

    def test_pipeline_handles_transcription_failure(self, tmp_path):
        from whisper_cli.scanner import scan_folder
        from whisper_cli.transcriber import transcribe

        videos_dir = tmp_path / "videos"
        videos_dir.mkdir()
        (videos_dir / "corrupt.mp4").write_bytes(b"not a real video")

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("ffmpeg decode error")
        _mock_whisper_module.load_model.return_value = mock_model

        base = output_base(videos_dir)
        sp = state_path(base)
        state = load_state(sp)
        videos = scan_folder(videos_dir)

        for v in videos:
            try:
                transcribe(v.path, "tiny")
            except RuntimeError:
                mark_processed(state, v, "error_transcribe", error="ffmpeg decode error")
                save_state(state, sp)

        reloaded = load_state(sp)
        entry = list(reloaded.processed.values())[0]
        assert entry.status == "error_transcribe"
        assert entry.error == "ffmpeg decode error"

        # Error entries are retried on next run
        videos2 = scan_folder(videos_dir)
        pending = get_unprocessed(videos2, reloaded)
        assert len(pending) == 1


class TestWatcherSnippety:
    """Test that whisper_watcher._update_snippety writes to Snippety CSV."""

    def test_update_snippety_writes_csv(self, tmp_path, monkeypatch):
        import importlib
        import sys

        csv_path = tmp_path / "snippets.csv"

        # Patch load_config to return a config pointing at our tmp csv
        from whisper_cli.config import Config
        fake_cfg = Config(openai_api_key="fake", snippety_csv_path=csv_path)

        import whisper_watcher as ww
        monkeypatch.setattr(
            "whisper_watcher._update_snippety.__code__",
            ww._update_snippety.__code__,
        )

        # Directly exercise the logic with a patched load_config
        with patch("whisper_cli.config.load_config", return_value=fake_cfg):
            import importlib as il
            # Reload to pick up the patch in module scope
            video_path = tmp_path / "demo_talk.mp4"
            video_path.write_bytes(b"fake")

            # Import the actual function
            from whisper_watcher import _update_snippety
            _update_snippety(video_path, "- Key insight from the talk")

        import csv as csv_mod
        with open(csv_path, newline="") as f:
            rows = list(csv_mod.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["keyword"] == "vid-demo_talk"
        assert "Key insight" in rows[0]["content"]

    def test_update_snippety_skips_when_no_path(self, tmp_path, monkeypatch):
        """No CSV path configured → function returns silently, no file created."""
        from whisper_cli.config import Config
        fake_cfg = Config(openai_api_key="fake", snippety_csv_path=None)

        with patch("whisper_cli.config.load_config", return_value=fake_cfg):
            from whisper_watcher import _update_snippety
            video_path = tmp_path / "clip.mp4"
            _update_snippety(video_path, "some summary")

        # No CSV should have been created
        assert not list(tmp_path.glob("*.csv"))
