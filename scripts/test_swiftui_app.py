"""Relocate the SwiftUI app and check it is self-contained, signed and transcribes offline.

Requires the selected Whisper model to already exist in the local model cache.
Counterpart of test_standalone.py (Qt app) for the bundle built by build_swiftui_app.py.
"""
import argparse
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import tempfile

MACHO_MAGIC = (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xfe\xed\xfa\xcf")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("app", type=Path)
parser.add_argument("audio", type=Path)
parser.add_argument("--model", default="mlx-community/whisper-large-v3-mlx")  # the app default
args = parser.parse_args()

with tempfile.TemporaryDirectory(prefix="transcription déplacée ") as directory:
    root = Path(directory).resolve()
    app = root / "Mp4ToTranscript.app"
    subprocess.run(["ditto", str(args.app), str(app)], check=True)

    # Signature still valid after relocation (ad hoc or Developer ID).
    subprocess.run(["codesign", "--verify", "--deep", "--strict", str(app)], check=True)

    info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
    assert info["LSMinimumSystemVersion"] == "26.0", info
    assert (app / "Contents/Resources" / f"{info['CFBundleIconFile']}.icns").is_file()
    types = info["CFBundleDocumentTypes"][0]["LSItemContentTypes"]
    assert {"public.audio", "public.movie"} <= set(types), types

    native_files = []
    for path in app.rglob("*"):
        if path.is_symlink():
            assert path.resolve().is_relative_to(app), f"External symlink: {path}"
        elif path.is_file():
            with path.open("rb") as stream:
                if stream.read(4) in MACHO_MAGIC:
                    native_files.append(str(path))
    dependencies = subprocess.run(["/usr/bin/otool", "-L", *native_files],
                                  capture_output=True, text=True, check=True).stdout
    external = [line for line in dependencies.splitlines()
                if line.startswith("\t") and line.strip().startswith("/")
                and not line.strip().startswith(("/usr/lib/", "/System/Library/"))]
    assert not external, f"Unbundled native dependencies: {external}"
    assert not any("Qt" in Path(f).name for f in native_files), "Qt must not be bundled"

    helper = app / "Contents/Helpers/Mp4ToTranscriptWorker.app"
    worker = helper / "Contents/MacOS/mp4-transcript-worker"
    env = dict(os.environ, PATH="/usr/bin:/bin", HF_HUB_OFFLINE="1",
               PYTHONPATH="", PYTHONHOME="/nonexistent-python")
    for tool in ("ffmpeg", "ffprobe"):
        subprocess.run([str(helper / "Contents/Frameworks/bin" / tool), "-version"],
                       cwd=root, env=env, check=True, capture_output=True, timeout=30)
    request = {"source": str(args.audio.resolve()), "cache_dir": str(root / "cache"),
               "options": {"model": args.model, "language": "fr", "prompt": "",
                           "format": "txt", "mode": "clean", "timestamps": False,
                           "output": str(root / "exports"), "cloud": False, "api_key": ""}}
    result = subprocess.run([str(worker)], input=json.dumps(request) + "\n",
                            text=True, capture_output=True, cwd=root, env=env, timeout=300)
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
    size = subprocess.run(["du", "-sh", str(app)], capture_output=True, text=True).stdout.split()[0]
    print(f"PASS: {size}, {len(native_files)} native binaries audited, signature valid after "
          "relocation, no Qt, bundled FFmpeg/FFprobe, offline transcription.")
