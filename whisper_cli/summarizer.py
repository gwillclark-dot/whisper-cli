import time

from openai import OpenAI

SYSTEM_PROMPT = (
    "You are a transcript summarizer. Your job is to extract the most important "
    "information from a video transcript, using the speaker's **exact words** wherever possible.\n\n"
    "Rules:\n"
    "- Lead each bullet with a direct quote from the transcript in double quotes, then a dash, then brief context if needed.\n"
    "- Only paraphrase when no quotable phrase exists for that point.\n"
    "- Cover: key decisions, action items, notable claims, and any named people/numbers/dates.\n"
    "- Format: bullet points only. No intro sentence. No conclusion.\n"
    "- Max 200 words total."
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
