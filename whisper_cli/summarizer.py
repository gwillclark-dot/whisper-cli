import time

from openai import OpenAI

SYSTEM_PROMPT = (
    "You are a helpful assistant. Summarize the following video transcript "
    "into a concise, scannable summary. Include: main topics discussed, "
    "key points or decisions, and any action items. Format with bullet points. "
    "Keep it under 200 words."
)

MAX_TRANSCRIPT_CHARS = 120_000


def summarize(transcript: str, filename: str, api_key: str) -> str:
    client = OpenAI(api_key=api_key)

    text = transcript
    if len(text) > MAX_TRANSCRIPT_CHARS:
        text = text[:MAX_TRANSCRIPT_CHARS] + "\n\n[Transcript truncated]"

    user_msg = f"Video: {filename}\n\nTranscript:\n{text}"

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=500,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
