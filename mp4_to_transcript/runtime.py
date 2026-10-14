"""Paths shared by the source checkout and the self-contained application."""
import os
import sys
from pathlib import Path


def configure_runtime():
    if getattr(sys, "frozen", False):
        binary_dir = Path(sys._MEIPASS) / "bin"
        os.environ["PATH"] = str(binary_dir) + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin")


def worker_arguments():
    return ["--worker"] if getattr(sys, "frozen", False) else ["-u", "-m", "mp4_to_transcript.desktop_worker"]
