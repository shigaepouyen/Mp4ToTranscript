"""PyInstaller entry point. Worker mode never initializes the GUI."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from mp4_to_transcript.runtime import configure_runtime
    configure_runtime()
    if "--worker" in sys.argv:
        from mp4_to_transcript.desktop_worker import main
    else:
        if "--smoke-test" not in sys.argv:
            from pathlib import Path
            log = Path.home() / "Library/Logs/Mp4ToTranscript/launch.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            sys.stdout = sys.stderr = log.open("a", buffering=1)
        from mp4_to_transcript.desktop import main
    raise SystemExit(main())
