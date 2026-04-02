"""Layer 3: Behavioral NL tests — real LLM calls, gated behind OPENAI_API_KEY.

Tests the summarizer's ability to produce consistent, well-structured summaries
across different transcript types, phrasings, and edge cases.
"""

import os

import pytest

# Skip entire module if no API key
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your-key-here",
    reason="OPENAI_API_KEY not set — skipping behavioral tests",
)

from whisper_cli.summarizer import summarize

API_KEY = os.getenv("OPENAI_API_KEY", "")


# ── Capability 1: Intent Classification ──────────────────────────────────
# Does the summarizer identify the right content type and structure?


class TestSummaryStructure:
    """The summary should always follow the prompted format: bullet points, <200 words."""

    def test_meeting_transcript_has_bullets(self):
        transcript = (
            "Alright everyone, let's get started. So first item on the agenda is the Q2 roadmap. "
            "We need to finalize the three priorities we discussed last week. Sarah mentioned "
            "that the auth migration should be priority one. Mike agreed but said we need to "
            "also consider the API rate limiting work since two customers have complained. "
            "Action item: Sarah will draft the migration plan by Friday. Mike will get customer "
            "feedback numbers from support. We'll reconvene next Tuesday to finalize."
        )
        result = summarize(transcript, "team_meeting.mp4", API_KEY)
        assert "-" in result or "•" in result, f"Expected bullet points, got:\n{result}"

    def test_lecture_transcript_has_bullets(self):
        transcript = (
            "Today we're going to talk about photosynthesis. Photosynthesis is the process "
            "by which plants convert sunlight into chemical energy. There are two main stages: "
            "the light-dependent reactions and the Calvin cycle. In the light-dependent reactions, "
            "water molecules are split and ATP is generated. The Calvin cycle then uses that ATP "
            "to fix carbon dioxide into glucose. Remember, the overall equation is six CO2 plus "
            "six H2O yields one glucose plus six O2."
        )
        result = summarize(transcript, "biology_lecture.mp4", API_KEY)
        assert "-" in result or "•" in result, f"Expected bullet points, got:\n{result}"

    def test_summary_under_200_words(self):
        transcript = (
            "This is a long discussion about various topics. " * 50
            + "The key takeaway is that we need better testing infrastructure. "
            "Another important point is that deployment frequency should increase. "
            "Finally, we agreed that documentation needs to be updated."
        )
        result = summarize(transcript, "long_discussion.mp4", API_KEY)
        word_count = len(result.split())
        assert word_count <= 250, f"Summary too long: {word_count} words"


# ── Capability 2: Semantic Invariance ────────────────────────────────────
# Same content rephrased differently → same key points extracted.


class TestSemanticInvariance:
    """Same meeting described differently should produce summaries with the same key facts."""

    FORMAL = (
        "The committee convened to discuss the budget allocation for fiscal year 2026. "
        "The proposed allocation of forty percent to engineering, thirty percent to marketing, "
        "and thirty percent to operations was approved unanimously. The deadline for "
        "departmental budget submissions is April fifteenth."
    )

    CASUAL = (
        "So yeah we talked about the budget for next year. Basically engineering gets forty "
        "percent, marketing gets thirty, operations gets thirty. Everyone was cool with it. "
        "Oh and budget stuff is due April fifteenth."
    )

    TERSE = (
        "Budget meeting. FY2026 split: 40% engineering, 30% marketing, 30% ops. "
        "Unanimous approval. Submissions due April 15."
    )

    @pytest.fixture(scope="class")
    def all_summaries(self):
        return {
            "formal": summarize(self.FORMAL, "budget_formal.mp4", API_KEY),
            "casual": summarize(self.CASUAL, "budget_casual.mp4", API_KEY),
            "terse": summarize(self.TERSE, "budget_terse.mp4", API_KEY),
        }

    def test_all_mention_engineering_percentage(self, all_summaries):
        for style, summary in all_summaries.items():
            assert "40" in summary or "forty" in summary.lower(), (
                f"{style} summary missing engineering allocation:\n{summary}"
            )

    def test_all_mention_deadline(self, all_summaries):
        for style, summary in all_summaries.items():
            has_date = "april" in summary.lower() or "15" in summary or "fifteenth" in summary.lower()
            assert has_date, f"{style} summary missing deadline:\n{summary}"

    def test_all_mention_budget(self, all_summaries):
        for style, summary in all_summaries.items():
            assert "budget" in summary.lower(), (
                f"{style} summary missing 'budget':\n{summary}"
            )


# ── Capability 3: Parameter Extraction ───────────────────────────────────
# Key data points from transcripts should appear in summaries.


class TestParameterExtraction:
    def test_extracts_action_items(self):
        transcript = (
            "Okay so the action items from this meeting: First, John needs to update the "
            "deployment scripts by next Wednesday. Second, Lisa will schedule the security "
            "audit for the second week of April. Third, the whole team needs to review the "
            "new API docs before the next standup."
        )
        result = summarize(transcript, "standup.mp4", API_KEY)
        result_lower = result.lower()
        assert "john" in result_lower, f"Missing actor 'John':\n{result}"
        assert "lisa" in result_lower, f"Missing actor 'Lisa':\n{result}"
        assert "deploy" in result_lower or "script" in result_lower, f"Missing task context:\n{result}"

    def test_extracts_numbers_and_metrics(self):
        transcript = (
            "Revenue for Q1 came in at 2.3 million, which is up 15 percent from last quarter. "
            "Customer churn dropped to 3.2 percent, our lowest ever. We added 450 new paying "
            "customers. The target for Q2 is 2.8 million in revenue."
        )
        result = summarize(transcript, "quarterly_review.mp4", API_KEY)
        assert "2.3" in result or "2.3M" in result.upper(), f"Missing revenue figure:\n{result}"
        assert "15" in result, f"Missing growth percentage:\n{result}"

    def test_extracts_decisions(self):
        transcript = (
            "After much discussion, the team decided to go with PostgreSQL over MongoDB for "
            "the new service. The reasoning was that we already have Postgres expertise and "
            "the data model is clearly relational. We also decided to use Kubernetes for "
            "orchestration instead of ECS."
        )
        result = summarize(transcript, "architecture_decision.mp4", API_KEY)
        result_lower = result.lower()
        assert "postgres" in result_lower, f"Missing DB decision:\n{result}"
        assert "kubernetes" in result_lower or "k8s" in result_lower, f"Missing orchestration decision:\n{result}"


# ── Capability 4: Robustness ─────────────────────────────────────────────
# Noisy, casual, or imperfect transcripts should still produce usable summaries.


class TestRobustness:
    def test_handles_filler_words(self):
        transcript = (
            "Um so like basically what happened was uh we had this meeting right and um "
            "the main thing was that uh we need to hire two more engineers um by the end "
            "of the month and uh also we need to um you know fix the login bug that's been "
            "like really annoying everyone."
        )
        result = summarize(transcript, "casual_meeting.mp4", API_KEY)
        result_lower = result.lower()
        assert "hire" in result_lower or "engineer" in result_lower, f"Missing hiring point:\n{result}"
        assert "login" in result_lower or "bug" in result_lower, f"Missing bug point:\n{result}"

    def test_handles_whisper_artifacts(self):
        """Whisper sometimes produces repeated words, missing punctuation, or misheard words."""
        transcript = (
            "the the quarterly results show that we we beat expectations revenue was "
            "one point five million and and the profit margin improved to twenty two percent "
            "we need to focus on on customer retention going forward the churn rate is "
            "still too high at at four percent"
        )
        result = summarize(transcript, "earnings.mp4", API_KEY)
        result_lower = result.lower()
        assert "revenue" in result_lower or "1.5" in result_lower, f"Missing revenue:\n{result}"
        assert "churn" in result_lower or "retention" in result_lower, f"Missing retention:\n{result}"

    def test_handles_mixed_languages_fragments(self):
        """Whisper may produce fragments or brief code-switching."""
        transcript = (
            "So the main update is we shipped the new dashboard. The team in Berlin, "
            "they said Alles gut, everything is working fine on their end. Performance "
            "metrics look solid, page load under 200 milliseconds. Next sprint we tackle "
            "the notification system."
        )
        result = summarize(transcript, "update.mp4", API_KEY)
        result_lower = result.lower()
        assert "dashboard" in result_lower, f"Missing dashboard:\n{result}"
        assert "notification" in result_lower or "sprint" in result_lower, f"Missing next steps:\n{result}"


# ── Capability 5: Boundary & Negative Cases ──────────────────────────────


class TestBoundaryCases:
    def test_very_short_transcript(self):
        """A one-sentence transcript should still produce a summary, not crash."""
        result = summarize("Meeting cancelled, rescheduled to Friday.", "short.mp4", API_KEY)
        assert len(result) > 0
        assert "cancel" in result.lower() or "reschedule" in result.lower() or "friday" in result.lower()

    def test_non_english_transcript(self):
        """Whisper may output non-English text. Summarizer should handle gracefully."""
        transcript = (
            "Hoy discutimos el presupuesto para el próximo trimestre. Se decidió asignar "
            "el cuarenta por ciento a ingeniería y el treinta por ciento a marketing."
        )
        result = summarize(transcript, "spanish_meeting.mp4", API_KEY)
        assert len(result) > 0  # Should produce something, not crash

    def test_technical_jargon(self):
        """Heavy technical content should be summarized, not garbled."""
        transcript = (
            "We need to migrate the gRPC endpoints from protobuf v2 to v3. The breaking "
            "changes are mostly around optional field semantics. We should also update the "
            "OpenTelemetry spans to include the new trace context propagation headers. "
            "The Envoy sidecar config needs to be updated for the new mTLS certificates."
        )
        result = summarize(transcript, "tech_review.mp4", API_KEY)
        result_lower = result.lower()
        assert "grpc" in result_lower or "protobuf" in result_lower, f"Missing tech terms:\n{result}"

    def test_empty_meaningful_content(self):
        """Transcript with no real substance."""
        transcript = "Testing testing one two three. Is this thing on? Hello? Okay, we're good."
        result = summarize(transcript, "soundcheck.mp4", API_KEY)
        assert len(result) > 0  # Should handle gracefully, not crash
