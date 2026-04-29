import os

for _malloc_env_key in (
    "MallocStackLogging",
    "MallocStackLoggingNoCompact",
    "MallocScribble",
    "MallocPreScribble",
    "MallocGuardEdges",
):
    os.environ.pop(_malloc_env_key, None)

from mp4_to_transcript.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
