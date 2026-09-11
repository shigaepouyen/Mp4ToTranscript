"""Build a local-use .app launcher with its own copy of the application source.

Uses the current Python environment, which must include .[desktop]. Rebuild after
moving/removing that environment. This is not a signed standalone distribution.
"""
import argparse
import plistlib
import shlex
import shutil
import subprocess
import sys
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
    launcher = macos / "Mp4ToTranscript"
    launcher.write_text('#!/bin/sh\n'
                        'cd "$(dirname "$0")/../Resources" || exit 1\n'
                        'export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"\n'
                        'log_dir="$HOME/Library/Logs/Mp4ToTranscript"\n'
                        'mkdir -p "$log_dir"\n'
                        f'{shlex.quote(python)} -m mp4_to_transcript.desktop "$@" >"$log_dir/launch.log" 2>&1\n'
                        'result=$?\n'
                        'if [ "$result" -ne 0 ]; then\n'
                        '  /usr/bin/open -a TextEdit "$log_dir/launch.log"\n'
                        'fi\n'
                        'exit "$result"\n')
    launcher.chmod(0o755)
    with (contents / "Info.plist").open("wb") as stream:
        plistlib.dump({"CFBundleName": "Mp4ToTranscript", "CFBundleDisplayName": "Mp4ToTranscript",
                      "CFBundleIdentifier": "io.github.shigaepouyen.mp4totranscript",
                      "CFBundleVersion": "1", "CFBundleShortVersionString": "0.2.0",
                      "CFBundleExecutable": "Mp4ToTranscript", "CFBundlePackageType": "APPL",
                      "NSHighResolutionCapable": True}, stream)
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("dist"))
    args = parser.parse_args()
    print(build(args.destination.resolve()))
