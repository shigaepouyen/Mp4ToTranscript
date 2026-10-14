# The JSON-line worker alone, embedded in the SwiftUI app as Contents/Helpers/Mp4ToTranscriptWorker.app.
# A nested bundle (not a bare one-folder) so codesign accepts the *.dist-info folders.
from pathlib import Path
import shutil
import tomllib
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata

root = Path(SPECPATH).parent
version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
datas = collect_data_files("mlx_whisper") + collect_data_files("mlx", includes=["lib/*.metallib"])
for package in ("mlx-whisper", "mlx", "huggingface-hub", "tiktoken", "regex", "numpy"):
    datas += copy_metadata(package)
binaries = collect_dynamic_libs("mlx")
for program in ("ffmpeg", "ffprobe"):
    executable = shutil.which(program)
    if not executable:
        raise RuntimeError(f"Install {program} before building")
    binaries.append((executable, "bin"))

a = Analysis([str(root / "scripts/worker_entry.py")], pathex=[str(root)],
    binaries=binaries, datas=datas,
    # MLX's native extension imports these Python helpers dynamically.
    hiddenimports=["mlx_whisper", "mlx._reprlib_fix", "mlx.__array_api_info", "tiktoken_ext.openai_public"],
    excludes=["PySide6", "shiboken6", "torch", "torchaudio", "pyannote", "tkinter",
              "matplotlib", "pandas", "IPython", "pytest"],
    noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="mp4-transcript-worker",
    debug=False, strip=False, upx=False, console=True,
    argv_emulation=False, target_arch="arm64", codesign_identity=None,
    entitlements_file=None)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="mp4-transcript-worker")
app = BUNDLE(coll, name="Mp4ToTranscriptWorker.app",
    bundle_identifier="io.github.shigaepouyen.mp4totranscript.worker",
    info_plist={"LSBackgroundOnly": True, "LSMinimumSystemVersion": "26.0",
                "CFBundleShortVersionString": version, "CFBundleVersion": version})
