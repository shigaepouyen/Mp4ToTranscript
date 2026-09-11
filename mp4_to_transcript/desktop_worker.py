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


def status(step: int, text: str, detail: str = "") -> None:
    emit("status", step=step, text=text, detail=detail)


def complete_model(path: Path) -> bool:
    return (path / "config.json").is_file() and any(
        (path / name).is_file() and (path / name).stat().st_size > 0
        for name in ("weights.safetensors", "weights.npz"))


def resolve_model(model: str) -> str:
    """Return a usable local directory; cached models never invoke the network."""
    status(2, "Vérification du modèle", "Recherche des fichiers du modèle sur votre Mac…")
    if complete_model(Path(model)):
        local = Path(model)
    else:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        try:
            local = Path(snapshot_download(repo_id=model, local_files_only=True))
        except LocalEntryNotFoundError:
            local = None
        if local is None or not complete_model(local):
            emit("model", text="Modèle absent ou incomplet : téléchargement nécessaire.")
            status(2, "Téléchargement du modèle", "Téléchargement depuis Hugging Face. Il sera réutilisé aux prochains lancements.")
            with contextlib.redirect_stdout(sys.stderr):
                local = Path(snapshot_download(repo_id=model,
                    allow_patterns=["config.json", "weights.safetensors", "weights.npz"]))
            if not complete_model(local):
                raise RuntimeError("Le téléchargement du modèle est incomplet. Réessayez lorsque la connexion est disponible.")
            emit("model", text="Modèle téléchargé et enregistré sur votre Mac.")
            return str(local)
    emit("model", text="Modèle déjà présent sur ce Mac · aucun téléchargement du modèle.")
    return str(local)


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
    status(1, "Vérification du fichier", "Lecture de l’empreinte du fichier et recherche d’une transcription existante…")
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
        model_path = resolve_model(options["model"])
        status(3, "Préparation du moteur", "Initialisation de Whisper et du calcul local sur le Mac…")
        with contextlib.redirect_stdout(sys.stderr):
            cli.load_whisper_module()
        status(4, "Transcription de l’audio", "Chargement du modèle en mémoire, puis analyse de l’audio. Cette étape peut prendre plusieurs minutes.")
        with contextlib.redirect_stdout(sys.stderr):
            result = cli.run_whisper(model_path, source, options["language"] or None,
                                     options["prompt"] or None, 0.0, "mlx")
        save_cache(cache, result)
    else:
        emit("model", text="Transcription déjà en cache · aucun modèle à charger ou télécharger.")
        status(4, "Transcription réutilisée", "Le texte existant est prêt : l’analyse audio est ignorée.")
    formats = ["txt", "md"] if options["format"] == "both" else [options["format"]]
    directory = Path(options["output"]) if options["output"] else source.parent / "transcripts"
    paths = []
    for fmt in formats:
        status(5, "Création du compte rendu" if options["mode"] == "meeting-plus" else "Création du fichier texte",
               "Enrichissement du texte avec OpenAI, puis enregistrement…" if options.get("cloud") else "Mise en forme et enregistrement sur votre Mac…")
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
