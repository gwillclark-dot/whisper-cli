import csv
from pathlib import Path

VIDEO_KEYWORD_PREFIX = "vid-"


def _read_existing_csv(csv_path: Path) -> list[dict[str, str]]:
    """Read existing Snippety CSV rows, return list of {keyword, title, content}."""
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "keyword": row.get("keyword", ""),
                "title": row.get("title", ""),
                "content": row.get("content", ""),
            })
    return rows


def export_snippets_csv(
    summaries: dict[str, str],
    csv_path: Path,
) -> None:
    """Merge video summaries into the Snippety CSV, preserving all non-video snippets.

    - Reads existing CSV (user's exported Snippety snippets)
    - Keeps all rows that don't have the vid- keyword prefix (user's other snippets)
    - Adds/updates rows for each video summary
    - Writes back the merged result
    """
    # Read existing rows, keep non-video ones
    existing = _read_existing_csv(csv_path)
    non_video_rows = [r for r in existing if not r["keyword"].startswith(VIDEO_KEYWORD_PREFIX)]

    # Build video rows from summaries
    video_rows = []
    for stem, summary in sorted(summaries.items()):
        video_rows.append({
            "keyword": f"{VIDEO_KEYWORD_PREFIX}{stem}",
            "title": f"Video: {stem}",
            "content": summary,
        })

    # Merge: user snippets first, then video summaries
    merged = non_video_rows + video_rows

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["keyword", "title", "content"])
        writer.writeheader()
        writer.writerows(merged)
