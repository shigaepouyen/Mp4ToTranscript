"""PyInstaller entry point for the SwiftUI app's helper: the JSON-line worker only, no Qt."""
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from mp4_to_transcript.runtime import configure_runtime
    configure_runtime()
    from mp4_to_transcript.desktop_worker import main
    raise SystemExit(main())
