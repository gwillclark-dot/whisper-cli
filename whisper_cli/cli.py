import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from whisper_cli.clipper import clip_video, parse_notes
from whisper_cli.config import Config, check_ffmpeg, load_config
from whisper_cli.scanner import scan_folder
from whisper_cli.state import (
    get_unprocessed,
    load_state,
    mark_processed,
    output_base,
    save_state,
    state_path,
)
from whisper_cli.summarizer import summarize
from whisper_cli.snippety import export_snippets_csv
from whisper_cli.transcriber import transcribe

app = typer.Typer(help="Transcribe videos, summarize, export to Snippety.")
console = Console()


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}TB"


def _setup_output(base: Path) -> tuple[Path, Path]:
    t = base / "transcriptions"
    s = base / "summaries"
    t.mkdir(parents=True, exist_ok=True)
    s.mkdir(parents=True, exist_ok=True)
    return t, s


def _update_snippety(cfg: Config, summaries_dir: Path) -> None:
    if not cfg.snippety_csv_path:
        console.print(
            "[yellow]SNIPPETY_CSV_PATH not set in .env — skipping Snippety update.[/yellow]\n"
            "[dim]  Add SNIPPETY_CSV_PATH=/path/to/file.csv to whisper-cli/.env[/dim]"
        )
        return

    summaries: dict[str, str] = {}
    for f in sorted(summaries_dir.iterdir()):
        if f.suffix == ".txt":
            summaries[f.stem] = f.read_text().strip()
    if not summaries:
        return

    export_snippets_csv(summaries, cfg.snippety_csv_path)
    console.print(f"[green]Snippety file updated:[/green] {cfg.snippety_csv_path}")


def _process_videos(
    folder: Path,
    model: str,
    no_summary: bool,
    no_snippet: bool,
    output: Path | None,
    dry_run: bool,
) -> int:
    base = output_base(folder, output)
    sp = state_path(base)
    state = load_state(sp)
    videos = scan_folder(folder)
    pending = get_unprocessed(videos, state)

    if not pending:
        console.print("[dim]No new videos.[/dim]")
        return 0

    if dry_run:
        console.print(f"[bold]{len(pending)} video(s) would be processed:[/bold]")
        for v in pending:
            console.print(f"  {v.path.name}  ({_human_size(v.size_bytes)})")
        return 0

    cfg = load_config(whisper_model=model)
    check_ffmpeg()

    trans_dir, summ_dir = _setup_output(base)
    console.print(f"[bold]{len(pending)} new video(s) to process[/bold]")
    console.print(f"[dim]Output → {base}[/dim]")
    errors = 0

    for i, v in enumerate(pending, 1):
        name = v.path.name
        stem = v.path.stem

        # Transcribe
        console.print(f"\n[cyan][{i}/{len(pending)}] Transcribing[/cyan] {name}  ({_human_size(v.size_bytes)})")
        try:
            transcript = transcribe(v.path, model)
        except Exception as e:
            console.print(f"[red]Transcription failed:[/red] {e}")
            mark_processed(state, v, "error_transcribe", error=str(e))
            save_state(state, sp)
            errors += 1
            continue

        # Save transcription
        trans_path = trans_dir / f"{stem}.txt"
        trans_path.write_text(transcript)
        console.print(f"  Saved: {trans_path}")

        # Summarize
        summary = ""
        if not no_summary:
            console.print(f"  [cyan]Summarizing...[/cyan]")
            try:
                summary = summarize(transcript, name, cfg.openai_api_key)
            except Exception as e:
                console.print(f"  [red]Summarization failed:[/red] {e}")
                mark_processed(state, v, "error_summarize", transcript_chars=len(transcript), error=str(e))
                save_state(state, sp)
                errors += 1
                continue

            summ_path = summ_dir / f"{stem}.txt"
            summ_path.write_text(summary)
            console.print(f"  Saved: {summ_path}")

        mark_processed(
            state, v, "ok",
            transcript_chars=len(transcript),
            summary_chars=len(summary),
        )
        save_state(state, sp)
        console.print(f"  [green]Done[/green]")

    # Update Snippety file in-place
    if not no_snippet and not no_summary:
        _update_snippety(cfg, summ_dir)

    console.print(f"\n[bold]Processed {len(pending) - errors}/{len(pending)} videos[/bold]")
    return errors


@app.command()
def run(
    folder: Path = typer.Argument(..., help="Folder containing videos"),
    model: str = typer.Option("base", "--model", "-m", help="Whisper model (tiny/base/small/medium/large)"),
    no_summary: bool = typer.Option(False, "--no-summary", help="Skip summarization"),
    no_snippet: bool = typer.Option(False, "--no-snippet", help="Skip Snippety update"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output dir (default: <folder>/vidsum-output/)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview what would be processed without doing it"),
):
    """Process all new videos in a folder (one-shot)."""
    folder = folder.resolve()
    if not folder.is_dir():
        raise typer.BadParameter(f"Not a directory: {folder}")
    _process_videos(folder, model, no_summary, no_snippet, output.resolve() if output else None, dry_run)


@app.command()
def watch(
    folder: Path = typer.Argument(..., help="Folder to watch for new videos"),
    model: str = typer.Option("base", "--model", "-m", help="Whisper model"),
    interval: int = typer.Option(30, "--interval", "-n", help="Poll interval in seconds"),
    no_summary: bool = typer.Option(False, "--no-summary"),
    no_snippet: bool = typer.Option(False, "--no-snippet"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output dir"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview only, don't process"),
):
    """Continuously watch a folder for new videos."""
    folder = folder.resolve()
    if not folder.is_dir():
        raise typer.BadParameter(f"Not a directory: {folder}")

    stop = False

    def _handle_signal(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    resolved_output = output.resolve() if output else None
    console.print(f"[bold]Watching[/bold] {folder} every {interval}s (Ctrl-C to stop)")

    while not stop:
        _process_videos(folder, model, no_summary, no_snippet, resolved_output, dry_run)
        now = datetime.now().strftime("%H:%M:%S")
        console.print(f"[dim]{now} — waiting {interval}s...[/dim]")
        for _ in range(interval):
            if stop:
                break
            time.sleep(1)

    console.print("\n[dim]Stopped.[/dim]")


@app.command("list")
def list_videos(
    folder: Path = typer.Argument(..., help="Folder to inspect"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output dir to read state from"),
):
    """Show processed and unprocessed videos."""
    folder = folder.resolve()
    base = output_base(folder, output.resolve() if output else None)
    sp = state_path(base)
    state = load_state(sp)
    videos = scan_folder(folder)
    pending = get_unprocessed(videos, state)

    table = Table(title=f"Videos in {folder}")
    table.add_column("File")
    table.add_column("Size")
    table.add_column("Status")
    table.add_column("Processed At")

    for v in videos:
        key = str(v.path)
        entry = state.processed.get(key)
        if entry:
            style = "green" if entry.status == "ok" else "red"
            table.add_row(
                v.path.name,
                _human_size(v.size_bytes),
                f"[{style}]{entry.status}[/{style}]",
                entry.processed_at,
            )
        else:
            table.add_row(v.path.name, _human_size(v.size_bytes), "[dim]pending[/dim]", "")

    console.print(table)
    if pending:
        console.print(f"\n[bold]{len(pending)} unprocessed video(s)[/bold]")


@app.command()
def reset(
    folder: Path = typer.Argument(..., help="Folder whose state to reset"),
    file: Optional[Path] = typer.Option(None, "--file", help="Reset only this file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output dir"),
):
    """Clear processing state to allow re-processing."""
    folder = folder.resolve()
    base = output_base(folder, output.resolve() if output else None)
    sp = state_path(base)
    state = load_state(sp)

    if file:
        key = str(file.resolve())
        if key in state.processed:
            del state.processed[key]
            save_state(state, sp)
            console.print(f"Reset: {file.name}")
        else:
            console.print(f"[yellow]Not found in state:[/yellow] {file}")
    else:
        if not state.processed:
            console.print("[dim]State already empty.[/dim]")
            return
        confirm = typer.confirm(f"Reset all {len(state.processed)} entries?")
        if confirm:
            state.processed.clear()
            save_state(state, sp)
            console.print("[green]State cleared.[/green]")


@app.command()
def clip(
    video: Path = typer.Argument(..., help="Video file to cut clips from"),
    notes: Path = typer.Argument(..., help="Notes file with timestamped clip specs (start-end label, one per line)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output dir for clips (default: <video_dir>/clips/)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview clips without running ffmpeg"),
):
    """Cut clips from a video using timestamped notes.

    Notes file format (one clip per line):
      MM:SS-MM:SS  label
      HH:MM:SS-HH:MM:SS  label

    Example:
      0:30-1:15  intro
      2:00-3:45  key point
      # comments and blank lines are ignored
    """
    video = video.resolve()
    notes = notes.resolve()

    if not video.is_file():
        raise typer.BadParameter(f"Video file not found: {video}")
    if not notes.is_file():
        raise typer.BadParameter(f"Notes file not found: {notes}")

    check_ffmpeg()

    notes_text = notes.read_text()
    specs = parse_notes(notes_text)
    if not specs:
        console.print("[yellow]No valid clip specs found in notes file.[/yellow]")
        console.print("[dim]Expected format: MM:SS-MM:SS label (one per line)[/dim]")
        raise typer.Exit(1)

    out_dir = output.resolve() if output else video.parent / "clips"

    console.print(f"[bold]{len(specs)} clip(s) from[/bold] {video.name}")
    console.print(f"[dim]Output → {out_dir}[/dim]")

    if dry_run:
        for i, s in enumerate(specs, 1):
            m, sec = divmod(int(s.start), 60)
            h, m = divmod(m, 60)
            start_fmt = f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"
            dur = s.end - s.start
            console.print(f"  [{i}] {start_fmt}  +{dur:.0f}s  {s.label}")
        return

    try:
        outputs = clip_video(video, notes, out_dir)
    except Exception as e:
        console.print(f"[red]Clip failed:[/red] {e}")
        raise typer.Exit(1)

    for i, (spec, path) in enumerate(zip(specs, outputs), 1):
        dur = spec.end - spec.start
        console.print(f"  [{i}] [green]✓[/green] {path.name}  ({dur:.0f}s)")

    console.print(f"\n[bold]Done — {len(outputs)} clip(s) saved to {out_dir}[/bold]")
