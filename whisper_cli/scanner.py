from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


@dataclass
class VideoFile:
    path: Path
    mtime: float
    size_bytes: int


def scan_folder(
    folder: Path,
    extensions: set[str] = VIDEO_EXTENSIONS,
) -> list[VideoFile]:
    videos = []
    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in extensions:
            stat = f.stat()
            videos.append(VideoFile(path=f.resolve(), mtime=stat.st_mtime, size_bytes=stat.st_size))
    videos.sort(key=lambda v: v.mtime)
    return videos
