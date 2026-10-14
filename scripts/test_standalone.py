"""Relocate a built app and transcribe real audio offline with bundled tools.

Requires the selected Whisper model to already exist in the local model cache.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("app", type=Path)
parser.add_argument("audio", type=Path)
parser.add_argument("--model", default="mlx-community/whisper-large-v3-turbo")
args = parser.parse_args()

with tempfile.TemporaryDirectory(prefix="transcription déplacée ") as directory:
    root = Path(directory).resolve()
    app = root / "Mp4ToTranscript.app"
    shutil.copytree(args.app, app, symlinks=True)
    native_files = []
    for path in app.rglob("*"):
        if path.is_symlink():
            assert path.resolve().is_relative_to(app), f"External symlink: {path}"
        elif path.is_file():
            with path.open("rb") as stream:
                magic = stream.read(4)
            if magic in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xfe\xed\xfa\xcf"):
                native_files.append(str(path))
    dependencies = subprocess.run(["/usr/bin/otool", "-L", *native_files],
                                  capture_output=True, text=True, check=True).stdout
    external = [line for line in dependencies.splitlines()
                if line.startswith("\t") and line.strip().startswith("/")
                and not line.strip().startswith(("/usr/lib/", "/System/Library/"))]
    assert not external, f"Unbundled native dependencies: {external}"
    binary = app / "Contents/MacOS/Mp4ToTranscript"
    env = dict(os.environ, PATH="/usr/bin:/bin", HF_HUB_OFFLINE="1",
               PYTHONPATH="", PYTHONHOME="/nonexistent-python",
               MP4_APP_SMOKE_REPORT=str(root / "smoke.json"))
    for tool in ("ffmpeg", "ffprobe"):
        subprocess.run([str(app / "Contents/Frameworks/bin" / tool), "-version"],
                       cwd=root, env=env, check=True, capture_output=True, timeout=30)
    subprocess.run([str(binary), "--smoke-test"], cwd=root, env=env,
                   check=True, capture_output=True, timeout=45)
    report = json.loads((root / "smoke.json").read_text())
    assert report["visible"] and report["icon_loaded"], report
    assert Path(report["python"]).is_relative_to(app), report
    request = {"source": str(args.audio.resolve()), "cache_dir": str(root / "cache"),
               "options": {"model": args.model, "language": "fr", "prompt": "",
                           "format": "txt", "mode": "clean", "timestamps": False,
                           "output": str(root / "exports"), "cloud": False, "api_key": ""}}
    result = subprocess.run([str(binary), "--worker"], input=json.dumps(request) + "\n",
                            text=True, capture_output=True, cwd=root, env=env, timeout=180)
    print(result.stdout, flush=True)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert any(e["event"] == "model" and "aucun téléchargement" in e["text"] for e in events)
    done = next(e for e in events if e["event"] == "done")
    for output in done["outputs"]:
        content = Path(output).read_text().strip()
        assert content, "Empty transcription"
        print(content)
    print(f"PASS: {len(native_files)} native binaries audited, relocated GUI, icon, bundled FFmpeg/FFprobe, offline transcription.")
