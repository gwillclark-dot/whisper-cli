"""Layer 2: Pipeline integration tests — mocked external services."""

import tempfile
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


class TestTranscriptionPipeline:
    """Test the transcription wrapper with mocked OpenAI Whisper API."""

    @patch("whisper_cli.transcriber.OpenAI")
    def test_transcribe_returns_stripped_text(self, mock_openai_cls, tmp_path):
        from whisper_cli.transcriber import transcribe

        video = tmp_path / "video.mp4"
        video.write_bytes(b"x" * 100)  # small fake file

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.audio.transcriptions.create.return_value = "  Hello world  "

        result = transcribe(video, api_key="fake-key")
        assert result == "Hello world"

    @patch("whisper_cli.transcriber.OpenAI")
    def test_transcribe_empty_result(self, mock_openai_cls, tmp_path):
        from whisper_cli.transcriber import transcribe

        video = tmp_path / "video.mp4"
        video.write_bytes(b"x" * 100)

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.audio.transcriptions.create.return_value = "   "

        result = transcribe(video, api_key="fake-key")
        assert result == ""

    def test_transcribe_rejects_oversized_file(self, tmp_path):
        from whisper_cli.transcriber import WHISPER_API_MAX_BYTES, transcribe

        video = tmp_path / "big.mp4"
        # Create a file that exceeds the 25MB limit
        video.write_bytes(b"x" * (WHISPER_API_MAX_BYTES + 1))

        with pytest.raises(ValueError, match="too large"):
            transcribe(video, api_key="fake-key")


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
    def test_summarize_truncates_long_transcript(self, mock_openai_cls):
        from whisper_cli.summarizer import MAX_TRANSCRIPT_CHARS, summarize

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = "summary"
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        long_text = "x" * (MAX_TRANSCRIPT_CHARS + 1000)
        summarize(long_text, "long.mp4", "fake-key")
        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert "[Transcript truncated]" in user_msg

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

    @patch("whisper_cli.summarizer.OpenAI")
    @patch("whisper_cli.transcriber.OpenAI")
    def test_full_pipeline_writes_outputs(self, mock_transcriber_cls, mock_openai_cls, tmp_path):
        from whisper_cli.scanner import scan_folder
        from whisper_cli.summarizer import summarize
        from whisper_cli.transcriber import transcribe

        # Setup video folder
        videos_dir = tmp_path / "videos"
        videos_dir.mkdir()
        (videos_dir / "meeting.mp4").write_bytes(b"fake video content")

        # Mock Whisper API
        mock_t_client = MagicMock()
        mock_transcriber_cls.return_value = mock_t_client
        mock_t_client.audio.transcriptions.create.return_value = "We discussed the Q2 roadmap and agreed on three priorities."

        # Mock OpenAI summarizer
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
            transcript = transcribe(v.path, api_key="fake-key")
            (trans_dir / f"{v.path.stem}.txt").write_text(transcript)

            summary = summarize(transcript, v.path.name, "fake-key")
            (summ_dir / f"{v.path.stem}.txt").write_text(summary)

            mark_processed(state, v, "ok", len(transcript), len(summary))
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

    @patch("whisper_cli.transcriber.OpenAI")
    def test_pipeline_handles_transcription_failure(self, mock_openai_cls, tmp_path):
        from whisper_cli.scanner import scan_folder
        from whisper_cli.transcriber import transcribe

        videos_dir = tmp_path / "videos"
        videos_dir.mkdir()
        (videos_dir / "corrupt.mp4").write_bytes(b"not a real video")

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.audio.transcriptions.create.side_effect = RuntimeError("API error")

        base = output_base(videos_dir)
        sp = state_path(base)
        state = load_state(sp)
        videos = scan_folder(videos_dir)

        for v in videos:
            try:
                transcribe(v.path, api_key="fake-key")
            except RuntimeError:
                mark_processed(state, v, "error_transcribe", error="API error")
                save_state(state, sp)

        reloaded = load_state(sp)
        entry = list(reloaded.processed.values())[0]
        assert entry.status == "error_transcribe"
        assert entry.error == "API error"

        # Error entries are retried on next run
        videos2 = scan_folder(videos_dir)
        pending = get_unprocessed(videos2, reloaded)
        assert len(pending) == 1
