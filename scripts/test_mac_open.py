"""Integration check: Launch Services cold/warm opens, using an isolated app ID."""
import json
import os
import plistlib
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from build_mac_app import build


with tempfile.TemporaryDirectory(prefix="mp4-dock-test-") as directory:
    root = Path(directory)
    app = build(root)
    info_path = app / "Contents/Info.plist"
    info = plistlib.loads(info_path.read_bytes())
    info["CFBundleIdentifier"] += ".test." + uuid.uuid4().hex
    info_path.write_bytes(plistlib.dumps(info))
    source1 = root / "premier fichier.wav"
    source2 = root / "deuxième fichier.mp4"
    source1.touch()
    source2.touch()
    report = root / "report.json"
    log = root / "launch.log"
    subprocess.run(["/usr/bin/open", "-n", "-a", str(app),
                    "--stdout", str(log), "--stderr", str(log),
                    "--env", f"MP4_APP_SMOKE_REPORT={report}",
                    "--env", "MP4_APP_SMOKE_DELAY_MS=8000",
                    str(source1), "--args", "--smoke-test"], check=True)
    # Wait for evidence of the first OS event before testing a warm open.
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if log.exists() and "Opened 1 file(s)" in log.read_text():
            break
        time.sleep(0.1)
    else:
        raise RuntimeError(log.read_text() if log.exists() else "App did not start")
    subprocess.run(["/usr/bin/open", "-a", str(app), str(source2)], check=True)
    while not report.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    if not report.exists():
        raise RuntimeError("No integration report: " + log.read_text())
    result = json.loads(report.read_text())
    assert result["visible"] and result["icon_loaded"], result
    # macOS can return decomposed Unicode filenames; compare filesystem identity.
    assert len(result["sources"]) == 2, result
    assert all(any(Path(received).samefile(expected) for received in result["sources"])
               for expected in (source1, source2)), result
    print(json.dumps(result, ensure_ascii=False))
