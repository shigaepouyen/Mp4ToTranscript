"""JSON-line worker for the desktop app. No Qt dependency; CLI stays unchanged."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from . import cli

CACHE_VERSION = 1


def emit(kind: str, **data) -> None:
    print(json.dumps({"event": kind, **data}, ensure_ascii=False), flush=True)


def cache_key(source: Path, options: dict) -> str:
    # Hash content, not just filename/mtime: replacing an audio must invalidate it.
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    digest.update(json.dumps({"v": CACHE_VERSION, "model": options["model"],
                              "language": options["language"],
                              "prompt": options["prompt"]}, sort_keys=True).encode())
    return digest.hexdigest()


def save_cache(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(result, stream, ensure_ascii=False,
                      default=lambda value: value.item() if hasattr(value, "item") else list(value))
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def write_unique(directory: Path, stem: str, suffix: str, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        candidate = directory / f"{stem}{'' if index == 1 else f' ({index})'}.{suffix}"
        try:
            with candidate.open("x", encoding="utf-8") as stream:
                stream.write(text)
            return candidate
        except FileExistsError:
            index += 1


def process(source: Path, options: dict, cache_dir: Path) -> list[str]:
    cli.ensure_ffmpeg_available()
    emit("status", text="Vérification du cache…")
    cache = cache_dir / f"{cache_key(source, options)}.json"
    result = None
    if cache.exists():
        try:
            result = json.loads(cache.read_text())
            if not isinstance(result, dict) or "segments" not in result:
                result = None
        except (ValueError, OSError):
            pass
    if result is None:
        emit("status", text="Transcription en cours…")
        with contextlib.redirect_stdout(sys.stderr):
            cli.load_whisper_module()
            result = cli.run_whisper(options["model"], source, options["language"] or None,
                                     options["prompt"] or None, 0.0, "mlx")
        save_cache(cache, result)
    else:
        emit("status", text="Transcription réutilisée · préparation du rendu…")
    formats = ["txt", "md"] if options["format"] == "both" else [options["format"]]
    directory = Path(options["output"]) if options["output"] else source.parent / "transcripts"
    paths = []
    for fmt in formats:
        emit("status", text="Création du compte rendu…" if options["mode"] == "meeting-plus" else "Export du texte…")
        with contextlib.redirect_stdout(sys.stderr):
            text = cli.render_transcription(result, source, options["timestamps"],
                                            options["mode"], fmt, False,
                                            llm_provider="openai" if options.get("cloud") else "none",
                                            openai_api_key=options.get("api_key") or None)
        paths.append(str(write_unique(directory, f"{source.stem} - {options['mode']}", fmt, text)))
    return paths


def main() -> int:
    try:
        request = json.loads(sys.stdin.readline())
        outputs = process(Path(request["source"]), request["options"], Path(request["cache_dir"]))
        emit("done", outputs=outputs)
        return 0
    except Exception as exc:
        detail = f"{exc}\n{exc.__cause__}" if exc.__cause__ else str(exc)
        emit("error", text=detail)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
