import json
import os
import tempfile
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

from mp4_to_transcript import desktop_worker as worker


def options():
    return dict(model="test-model", language="fr", prompt="", format="md", mode="clean",
                timestamps=False, output="", cloud=False, api_key="")


class WorkerTests(unittest.TestCase):
    def test_cache_reuses_transcription_for_new_render_but_invalidates_audio_and_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "réunion.mp4"
            source.write_bytes(b"audio")
            config = options()
            result = {"text": "Bonjour", "segments": [{"text": "Bonjour", "start": 0, "end": 1}]}
            with mock.patch.object(worker.cli, "ensure_ffmpeg_available"), mock.patch.object(worker.cli, "load_whisper_module"), mock.patch.object(worker.cli, "run_whisper", return_value=result) as transcribe, mock.patch.object(worker.cli, "render_transcription", return_value="Bonjour"):
                first = worker.process(source, config, root / "cache")
                config.update(mode="raw", timestamps=True)
                second = worker.process(source, config, root / "cache")
                self.assertEqual(transcribe.call_count, 1)
                self.assertNotEqual(first, second)
                self.assertEqual(Path(first[0]).read_text(), "Bonjour")
                config["prompt"] = "Nouveau contexte"
                worker.process(source, config, root / "cache")
                self.assertEqual(transcribe.call_count, 2)
                source.write_bytes(b"other")
                worker.process(source, config, root / "cache")
                self.assertEqual(transcribe.call_count, 3)

    def test_exports_never_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = worker.write_unique(root, "same", "md", "original")
            b = worker.write_unique(root, "same", "md", "new")
            self.assertNotEqual(a, b)
            self.assertEqual(a.read_text(), "original")

    def test_both_formats_only_transcribe_once_and_cloud_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "audio.wav"
            source.write_bytes(b"audio")
            config = options()
            config.update(format="both", output=str(root / "exports"))
            with mock.patch.object(worker.cli, "ensure_ffmpeg_available"), mock.patch.object(worker.cli, "load_whisper_module"), mock.patch.object(worker.cli, "run_whisper", return_value={"segments": []}) as transcribe, mock.patch.object(worker.cli, "render_transcription", return_value="Text") as render:
                outputs = worker.process(source, config, root / "cache")
                self.assertEqual(len(outputs), 2)
                self.assertEqual(transcribe.call_count, 1)
                self.assertTrue(all(call.kwargs["llm_provider"] == "none" for call in render.call_args_list))


try:
    from PySide6.QtCore import QSettings, QProcess
    from PySide6.QtWidgets import QApplication
    from mp4_to_transcript.desktop import Window
except ImportError:
    Window = None


@unittest.skipIf(Window is None, "Install .[desktop] for GUI tests")
class WindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_add_deduplicates_folders_retry_and_preferences_exclude_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test.mp4").touch()
            settings = QSettings(str(root / "prefs.ini"), QSettings.Format.IniFormat)
            window = Window(settings)
            window.add_paths([str(root), str(root / "test.mp4")])
            self.assertEqual(len(window.jobs), 1)
            self.assertTrue(window.start_button.isEnabled())
            window.jobs[0]["state"] = "Erreur"
            window.table.selectRow(0)
            window.retry_selected()
            self.assertEqual(window.jobs[0]["state"], "En attente")
            window.key.setText("secret-test-value")
            window.cloud.setChecked(True)
            window.close()
            settings.sync()
            self.assertNotIn("secret-test-value", (root / "prefs.ini").read_text())
            self.assertNotIn("cloud", settings.allKeys())

    def pump_until(self, predicate, timeout=8):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertTrue(predicate(), "Process did not reach expected state")

    def test_cancel_stops_worker_and_preserves_next_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["a.wav", "b.wav"]:
                (root / name).touch()
            window = Window(QSettings(str(root / "prefs.ini"), QSettings.Format.IniFormat))
            window.add_paths([str(root)])
            original_start = QProcess.start
            def fake_start(process, program, arguments):
                original_start(process, sys.executable, ["-u", "-c", "import sys,time; sys.stdin.readline(); time.sleep(30)"])
            with mock.patch.object(QProcess, "start", fake_start):
                window.start()
                self.pump_until(lambda: window.process.state() == QProcess.ProcessState.Running)
                window.cancel()
                self.pump_until(lambda: not window.running)
            self.assertEqual([j["state"] for j in window.jobs], ["Annulé", "En attente"])
            self.assertIsNone(window.process)
            self.assertTrue(window.start_button.isEnabled())
            window.close()

    def test_queue_continues_after_worker_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["a.wav", "b.wav"]:
                (root / name).touch()
            window = Window(QSettings(str(root / "prefs.ini"), QSettings.Format.IniFormat))
            window.add_paths([str(root)])
            original_start = QProcess.start
            def fake_start(process, program, arguments):
                original_start(process, sys.executable, ["-u", "-c", "import sys,json; sys.stdin.readline(); print(json.dumps({'event':'error','text':'Audio illisible'})); sys.exit(1)"])
            with mock.patch.object(QProcess, "start", fake_start):
                window.start()
                self.pump_until(lambda: not window.running)
            self.assertEqual([j["state"] for j in window.jobs], ["Erreur", "Erreur"])
            self.assertEqual(window.jobs[1]["error"], "Audio illisible")
            window.close()


if __name__ == "__main__":
    unittest.main()
