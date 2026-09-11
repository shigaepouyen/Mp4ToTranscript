"""Build a local-use .app launcher with its own copy of the application source.

Uses the current Python environment, which must include .[desktop]. Rebuild after
moving/removing that environment. This is not a signed standalone distribution.
"""
import argparse
import json
import plistlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def build(destination: Path, python: str = sys.executable) -> Path:
    # Fail during the build rather than creating a launcher that silently exits.
    subprocess.run([python, "-c", "from PySide6.QtWidgets import QApplication"], check=True)
    bundle = destination / "Mp4ToTranscript.app"
    contents = bundle / "Contents"
    resources = contents / "Resources"
    macos = contents / "MacOS"
    macos.mkdir(parents=True, exist_ok=True)
    resources.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[1] / "mp4_to_transcript"
    shutil.copytree(source, resources / "mp4_to_transcript", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(source / "assets" / "AppIcon.icns", resources / "AppIcon.icns")
    launcher = macos / "Mp4ToTranscript"
    runtime = json.loads(subprocess.check_output([python, "-c",
        "import sys, json; from pathlib import Path; "
        "print(json.dumps({'executable': sys.executable, 'library': str(Path(sys.base_prefix) / 'Python')}))"], text=True))
    if not Path(runtime["library"]).is_file():
        raise RuntimeError("A framework Python installation is required to build the macOS launcher.")
    with tempfile.TemporaryDirectory() as directory:
        header = Path(directory) / "runtime_config.h"
        header.write_text(f'#define PYTHON_EXECUTABLE {json.dumps(runtime["executable"])}\n'
                          f'#define PYTHON_LIBRARY {json.dumps(runtime["library"])}\n')
        subprocess.run(["/usr/bin/xcrun", "clang", "-O2", "-Wall", "-I", directory,
                        str(Path(__file__).with_name("mac_launcher.c")), "-o", str(launcher)], check=True)
    launcher.chmod(0o755)
    with (contents / "Info.plist").open("wb") as stream:
        plistlib.dump({"CFBundleName": "Mp4ToTranscript", "CFBundleDisplayName": "Mp4ToTranscript",
                      "CFBundleIdentifier": "io.github.shigaepouyen.mp4totranscript",
                      "CFBundleVersion": "3", "CFBundleShortVersionString": "0.3.0",
                      "CFBundleIconFile": "AppIcon.icns",
                      "CFBundleExecutable": "Mp4ToTranscript", "CFBundlePackageType": "APPL",
                      "NSHighResolutionCapable": True,
                      "CFBundleDocumentTypes": [{"CFBundleTypeName": "Audio, vidéo et dossiers",
                          "CFBundleTypeRole": "Viewer", "LSHandlerRank": "Alternate",
                          "LSItemContentTypes": ["public.audio", "public.movie", "public.folder"],
                          "CFBundleTypeExtensions": ["aac", "flac", "m4a", "m4v", "mkv", "mov", "mp3", "mp4", "mpeg", "mpga", "ogg", "opus", "wav", "webm", "wma"]}]}, stream)
    bundle.touch()
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("dist"))
    args = parser.parse_args()
    print(build(args.destination.resolve()))
