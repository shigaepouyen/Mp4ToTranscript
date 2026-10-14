"""Build an Apple Silicon app containing Python, Qt, MLX, and FFmpeg."""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def build_standalone(destination: Path, work: Path):
    root = Path(__file__).resolve().parents[1]
    work = work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYINSTALLER_CONFIG_DIR=str(work / "pyinstaller-cache"))
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm",
        "--distpath", str(destination.resolve()), "--workpath", str(work.resolve()),
        str(root / "scripts/Mp4ToTranscript.spec")], cwd=root, env=env, check=True)
    return destination.resolve() / "Mp4ToTranscript.app"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("dist"))
    parser.add_argument("--work", type=Path, default=Path("build"))
    args = parser.parse_args()
    print(build_standalone(args.destination, args.work))
