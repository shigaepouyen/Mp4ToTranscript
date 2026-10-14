# A relocatable one-folder macOS bundle. Models remain in Hugging Face's cache.
from pathlib import Path
import shutil
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata

root = Path(SPECPATH).parent
datas = collect_data_files("mlx_whisper") + collect_data_files("mlx", includes=["lib/*.metallib"])
datas += [(str(root / "mp4_to_transcript/assets"), "mp4_to_transcript/assets")]
for package in ("mlx-whisper", "mlx", "huggingface-hub", "tiktoken", "regex", "numpy"):
    datas += copy_metadata(package)
binaries = collect_dynamic_libs("mlx")
for program in ("ffmpeg", "ffprobe"):
    executable = shutil.which(program)
    if not executable:
        raise RuntimeError(f"Install {program} before building")
    binaries.append((executable, "bin"))

a = Analysis([str(root / "scripts/app_entry.py")], pathex=[str(root)],
    binaries=binaries, datas=datas,
    # MLX's native extension imports these Python helpers dynamically.
    hiddenimports=["mlx_whisper", "mlx._reprlib_fix", "mlx.__array_api_info", "tiktoken_ext.openai_public"],
    excludes=["torch", "torchaudio", "pyannote", "tkinter", "matplotlib", "pandas", "IPython", "pytest"],
    noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Mp4ToTranscript",
    debug=False, strip=False, upx=False, console=False,
    argv_emulation=False, target_arch="arm64", codesign_identity=None,
    entitlements_file=None)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="Mp4ToTranscript")
app = BUNDLE(coll, name="Mp4ToTranscript.app",
    icon=str(root / "mp4_to_transcript/assets/AppIcon.icns"),
    bundle_identifier="io.github.shigaepouyen.mp4totranscript",
    info_plist={
        "CFBundleDisplayName": "Mp4ToTranscript", "CFBundleShortVersionString": "0.5.0",
        "CFBundleVersion": "5", "NSHighResolutionCapable": True,
        "CFBundleDocumentTypes": [{"CFBundleTypeName": "Audio, vidéo et dossiers",
            "CFBundleTypeRole": "Viewer", "LSHandlerRank": "Alternate",
            "LSItemContentTypes": ["public.audio", "public.movie", "public.folder"],
            "CFBundleTypeExtensions": ["aac", "flac", "m4a", "m4v", "mkv", "mov", "mp3", "mp4", "mpeg", "mpga", "ogg", "opus", "wav", "webm", "wma"]}]})
