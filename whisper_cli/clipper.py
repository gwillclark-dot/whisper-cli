"""Video clip extraction from timestamped notes using ffmpeg."""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ClipSpec:
    start: float  # seconds
    end: float    # seconds
    label: str


def _parse_time(t: str) -> float:
    """Parse HH:MM:SS, MM:SS, or raw seconds string → float seconds."""
    t = t.strip()
    parts = t.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


# e.g. "0:30-1:15 intro" or "00:01:30-00:02:45 key moment"
_LINE_RE = re.compile(
    r"^([\d:]+\.?\d*)\s*[-–]\s*([\d:]+\.?\d*)\s+(.*?)\s*$"
)


def parse_notes(text: str) -> list[ClipSpec]:
    """Parse a notes file into ClipSpec list. Ignores blank lines and # comments."""
    specs: list[ClipSpec] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        start = _parse_time(m.group(1))
        end = _parse_time(m.group(2))
        label = m.group(3)
        if end > start:
            specs.append(ClipSpec(start=start, end=end, label=label))
    return specs


def _safe_filename(label: str) -> str:
    """Sanitize label for use as filename."""
    return re.sub(r"[^\w\-]", "_", label)[:60].strip("_") or "clip"


def cut_clip(
    video_path: Path,
    spec: ClipSpec,
    output_dir: Path,
    index: int,
    dry_run: bool = False,
) -> Path:
    """Extract one clip from video_path using ffmpeg. Returns output path."""
    stem = video_path.stem
    safe = _safe_filename(spec.label)
    out_name = f"{stem}_clip{index:02d}_{safe}{video_path.suffix}"
    out_path = output_dir / out_name

    duration = spec.end - spec.start
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(spec.start),
        "-i", str(video_path),
        "-t", str(duration),
        "-c", "copy",
        str(out_path),
    ]

    if dry_run:
        return out_path

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for clip '{spec.label}': {result.stderr[-500:]}")
    return out_path


def clip_video(
    video_path: Path,
    notes_path: Path,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> list[Path]:
    """Cut all clips defined in notes_path from video_path. Returns list of output paths."""
    notes_text = notes_path.read_text()
    specs = parse_notes(notes_text)
    if not specs:
        raise ValueError(f"No valid clip specs found in {notes_path}")

    out_dir = output_dir or video_path.parent / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for i, spec in enumerate(specs, 1):
        out = cut_clip(video_path, spec, out_dir, index=i, dry_run=dry_run)
        outputs.append(out)
    return outputs
