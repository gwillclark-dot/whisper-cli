import time
from typing import List

from openai import OpenAI

SYSTEM_PROMPT = """\
You are the brain behind a personal video capture tool. Someone saved this video \
because something in it was worth keeping — a technique, an idea, a quote, a \
reference, a feeling. Your job is to figure out WHAT was worth capturing and \
make it findable and useful later.

## Mindset
Think like a personal research assistant, not a summarizer. Ask yourself:
- Why would someone save this? What's the actionable nugget?
- Six months from now, what search term should find this summary?
- What would they want to copy-paste into a project, conversation, or note?

## Step 1 — Classify (do NOT output)
Silently determine: tutorial, interview, essay/rant, review, how-to, \
storytelling, pitch, demo, or conversation. This shapes emphasis.

## Step 2 — Output

**TL;DR:** One bold sentence — the single thing worth remembering. Use the \
speaker's words if they said it well.

**Why this matters:** 1-2 sentences on why this is worth saving. Connect it to \
a broader theme, skill, or decision. This is the "so what."

**Key takeaways:** Bullets grouped by topic if 3+ topics, flat otherwise.
- Lead with "direct quote" when a strong one exists — dash — context (5-15 words)
- For tutorials: capture steps, tools mentioned, and gotchas
- For interviews: capture each person's position and any disagreements
- For essays: the argument (claim → evidence → conclusion)
- For pitches/demos: the product, the differentiator, who it's for

**Best quotes:** 1-3 lines worth remembering on their own. Blockquote format:
> "quote here"

**Tags:** 3-6 lowercase keywords for search (comma-separated). Include: topic, \
domain, any named tools/people/frameworks, content type.

## Rules
- Pull ACTUAL phrases — never invent or embellish quotes.
- Include specific names, numbers, dates, URLs, tool names when they appear.
- Timestamps as (MM:SS) after quotes when available in the transcript.
- No meta-commentary ("this video discusses..."). No filler. Start with TL;DR.
- Target 200-400 words. Shorter is fine for thin content."""

CHUNK_SIZE = 40_000       # ~20 min of speech per chunk
CHUNK_OVERLAP = 500       # carry over sentence context between chunks
MAX_DIRECT_CHARS = 120_000  # below this, skip chunking

CHUNK_EXTRACT_PROMPT = """\
You are extracting key information from one segment of a longer video transcript.
Output ONLY a compact list — no narrative, no filler.

Format:
- Key points (bullet per idea, include timestamps if present as MM:SS)
- Notable quotes: "exact words" (context)
- Named tools/people/numbers/URLs mentioned

Keep it dense. 150 words max. This will feed into a final summary."""


def _chat(client: OpenAI, system: str, user: str, max_tokens: int = 800) -> str:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.3,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _split_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Try to break at a sentence boundary
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary + 2
        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else end
    return chunks


def summarize(transcript: str, filename: str, api_key: str) -> str:
    client = OpenAI(api_key=api_key)

    if len(transcript) <= MAX_DIRECT_CHARS:
        user_msg = f"Video: {filename}\n\nTranscript:\n{transcript}"
        return _chat(client, SYSTEM_PROMPT, user_msg)

    # Chunked path for long transcripts
    chunks = _split_chunks(transcript, CHUNK_SIZE, CHUNK_OVERLAP)
    extracts: List[str] = []
    for i, chunk in enumerate(chunks, 1):
        user_msg = f"Video: {filename} — segment {i}/{len(chunks)}\n\nTranscript segment:\n{chunk}"
        extract = _chat(client, CHUNK_EXTRACT_PROMPT, user_msg, max_tokens=300)
        extracts.append(f"[Segment {i}/{len(chunks)}]\n{extract}")

    combined = "\n\n".join(extracts)
    user_msg = (
        f"Video: {filename}\n\n"
        f"The following are extracted key points from {len(chunks)} segments of a long video. "
        f"Synthesize them into a single cohesive summary.\n\n{combined}"
    )
    return _chat(client, SYSTEM_PROMPT, user_msg)
