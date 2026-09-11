"""A small desktop front end; expensive work runs in a cancellable process."""
from __future__ import annotations

import codecs
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt, QProcess, QProcessEnvironment, QSettings, QTimer, QDir, QLibraryInfo
from PySide6.QtGui import QAction, QColor, QPalette, QIcon
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QSplitter, QSystemTrayIcon, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget, QHeaderView, QAbstractItemView,
    QStyle, QDialog,
)

from .cli import collect_input_files


class Window(QMainWindow):
    def __init__(self, settings=None):
        super().__init__()
        self.settings = settings or QSettings("Mp4ToTranscript", "Desktop")
        self.jobs = []
        self.process = None
        self.active = None
        self.running = False
        self.cancelled = False
        self.closing = False
        self.buffer = ""
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.logs = ""
        self.setWindowTitle("Mp4ToTranscript")
        self.resize(980, 730)
        self.setMinimumSize(820, 650)
        self.setAcceptDrops(True)
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(28, 24, 28, 22)
        layout.setSpacing(16)
        heading = QLabel("Vos enregistrements, en texte.")
        heading.setObjectName("heading")
        layout.addWidget(heading)
        self.drop = QLabel("Déposez vos fichiers ou dossiers ici\nAudio et vidéo · traitement sur votre Mac")
        self.drop.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop.setMinimumHeight(86)
        self.drop.setObjectName("drop")
        layout.addWidget(self.drop)
        buttons = QHBoxLayout()
        self.add_button = QPushButton("Ajouter des fichiers…")
        self.add_button.clicked.connect(self.choose_files)
        folder = QPushButton("Ajouter un dossier…")
        folder.clicked.connect(self.choose_folder)
        self.clear_button = QPushButton("Retirer la sélection")
        self.clear_button.clicked.connect(self.remove_selected)
        buttons.addWidget(self.add_button)
        buttons.addWidget(folder)
        buttons.addStretch()
        buttons.addWidget(self.clear_button)
        layout.addLayout(buttons)

        self.controls = QWidget()
        form = QHBoxLayout(self.controls)
        form.setContentsMargins(0, 0, 0, 0)
        self.profile = QComboBox()
        for label, value in [("Texte nettoyé", "clean"), ("Transcription brute", "raw"), ("Compte rendu", "meeting-plus")]:
            self.profile.addItem(label, value)
        self.language = QComboBox()
        for label, value in [("Français", "fr"), ("Détection automatique", ""), ("Anglais", "en")]:
            self.language.addItem(label, value)
        self.format = QComboBox()
        for label, value in [("Markdown (.md)", "md"), ("Texte (.txt)", "txt"), ("Les deux", "both")]:
            self.format.addItem(label, value)
        for label, widget, key in [("Profil", self.profile, "mode"), ("Langue", self.language, "language"), ("Format", self.format, "format")]:
            column = QVBoxLayout()
            field_label = QLabel(label)
            field_label.setBuddy(widget)
            column.addWidget(field_label)
            column.addWidget(widget)
            form.addLayout(column, 1)
            index = widget.findData(self.settings.value(key, widget.itemData(0)))
            widget.setCurrentIndex(max(0, index))
        layout.addWidget(self.controls)

        self.advanced_toggle = QPushButton("Réglages…")
        self.advanced_toggle.setFlat(True)
        self.advanced_toggle.setMaximumWidth(130)
        layout.addWidget(self.advanced_toggle)
        self.advanced = QDialog(self)
        self.advanced.setWindowTitle("Réglages")
        self.advanced.setWindowModality(Qt.WindowModality.WindowModal)
        self.advanced.setMinimumWidth(660)
        fields = QFormLayout(self.advanced)
        fields.setContentsMargins(22, 22, 22, 22)
        fields.setVerticalSpacing(10)
        self.timestamps = QCheckBox("Inclure les repères de temps")
        self.timestamps.setChecked(self.settings.value("timestamps", False, type=bool))
        fields.addRow("Repères", self.timestamps)
        self.model = QComboBox()
        for label, value in [("Précision · large v3", "mlx-community/whisper-large-v3-mlx"), ("Rapide · large v3 turbo", "mlx-community/whisper-large-v3-turbo"), ("Léger · medium", "mlx-community/whisper-medium-mlx")]:
            self.model.addItem(label, value)
        self.model.setCurrentIndex(max(0, self.model.findData(self.settings.value("model", self.model.itemData(0)))))
        fields.addRow("Modèle", self.model)
        self.prompt = QLineEdit(self.settings.value("prompt", ""))
        self.prompt.setPlaceholderText("Noms propres, jargon, contexte de la réunion…")
        fields.addRow("Contexte", self.prompt)
        self.output = QLineEdit(self.settings.value("output", ""))
        self.output.setPlaceholderText("Dossier transcripts à côté de chaque source")
        output_row = QHBoxLayout()
        output_row.addWidget(self.output)
        browse = QPushButton("Choisir…")
        browse.clicked.connect(self.choose_output)
        output_row.addWidget(browse)
        fields.addRow("Destination", output_row)
        self.cloud = QCheckBox("Enrichir avec OpenAI (compte rendu uniquement)")
        fields.addRow("Enrichissement", self.cloud)
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        self.key.setPlaceholderText("Clé API pour cette session, ou OPENAI_API_KEY")
        fields.addRow("Clé OpenAI", self.key)
        notice = QLabel("Si activé, le texte est envoyé à OpenAI et des frais API peuvent s’appliquer.\nLe mode local produit un compte rendu par règles, sans synthèse IA.")
        notice.setWordWrap(True)
        fields.addRow(notice)
        done = QPushButton("Terminé")
        done.clicked.connect(self.advanced.accept)
        fields.addRow("", done)
        self.advanced.hide()
        self.advanced_toggle.clicked.connect(self.advanced.show)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Fichier", "État"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.itemSelectionChanged.connect(self.preview_selected)
        splitter.addWidget(self.table)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setPlaceholderText("Le résultat du fichier sélectionné apparaîtra ici.")
        splitter.addWidget(self.preview)
        splitter.setSizes([180, 120])
        layout.addWidget(splitter, 1)
        actions = QHBoxLayout()
        self.copy_button = QPushButton("Copier le texte")
        self.copy_button.clicked.connect(self.copy_result)
        self.open_button = QPushButton("Afficher dans le Finder")
        self.open_button.clicked.connect(self.reveal_result)
        self.retry_button = QPushButton("Remettre en attente")
        self.retry_button.clicked.connect(self.retry_selected)
        actions.addWidget(self.copy_button)
        actions.addWidget(self.open_button)
        actions.addStretch()
        actions.addWidget(self.retry_button)
        layout.addLayout(actions)
        bottom = QHBoxLayout()
        self.status = QLabel("Ajoutez un enregistrement pour commencer.")
        self.status.setWordWrap(True)
        bottom.addWidget(self.status, 1)
        self.cancel_button = QPushButton("Arrêter la file")
        self.cancel_button.clicked.connect(self.cancel)
        bottom.addWidget(self.cancel_button)
        self.start_button = QPushButton("Transcrire")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self.start)
        bottom.addWidget(self.start_button)
        layout.addLayout(bottom)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(5)
        layout.addWidget(self.progress)
        self.tray = QSystemTrayIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaVolume), self)
        self.tray.setToolTip("Mp4ToTranscript")
        self.tray.show()
        shortcut = QAction("Ajouter des fichiers", self)
        shortcut.setShortcut("Ctrl+O")
        shortcut.triggered.connect(self.choose_files)
        self.addAction(shortcut)
        self.update_controls()

    def options(self):
        return {"mode": self.profile.currentData(), "language": self.language.currentData(),
                "format": self.format.currentData(), "model": self.model.currentData(),
                "timestamps": self.timestamps.isChecked(), "prompt": self.prompt.text().strip(),
                "output": str(Path(self.output.text().strip()).expanduser()) if self.output.text().strip() else "",
                "cloud": self.cloud.isChecked() and self.profile.currentData() == "meeting-plus",
                "api_key": self.key.text().strip()}

    def choose_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Ajouter des enregistrements", "", "Audio et vidéo (*.aac *.flac *.m4a *.m4v *.mkv *.mov *.mp3 *.mp4 *.mpeg *.mpga *.ogg *.opus *.wav *.webm *.wma);;Tous les fichiers (*)")
        self.add_paths(paths)

    def choose_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Ajouter un dossier (sous-dossiers inclus)")
        if path:
            self.add_paths([path])

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(self, "Dossier des résultats")
        if path:
            self.output.setText(path)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.add_paths([url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()])
        event.acceptProposedAction()

    def add_paths(self, paths):
        errors = []
        known = {job["source"] for job in self.jobs}
        for path in paths:
            try:
                for source in collect_input_files(Path(path).resolve(), recursive=True):
                    if str(source) not in known:
                        known.add(str(source))
                        self.jobs.append({"source": str(source), "state": "En attente", "outputs": [], "error": ""})
            except (OSError, ValueError) as exc:
                errors.append(str(exc))
        self.refresh()
        if errors:
            QMessageBox.warning(self, "Certains fichiers n’ont pas été ajoutés", "\n".join(errors))

    def refresh(self):
        self.table.setRowCount(len(self.jobs))
        for row, job in enumerate(self.jobs):
            name = QTableWidgetItem(Path(job["source"]).name)
            name.setToolTip(job["source"])
            self.table.setItem(row, 0, name)
            state = QTableWidgetItem(job["state"])
            state.setToolTip(job.get("error", ""))
            self.table.setItem(row, 1, state)
        self.update_controls()

    def selected(self):
        row = self.table.currentRow()
        return self.jobs[row] if 0 <= row < len(self.jobs) else None

    def update_controls(self):
        self.start_button.setEnabled(not self.running and any(j["state"] == "En attente" for j in self.jobs))
        self.cancel_button.setEnabled(self.running and not self.cancelled)
        self.clear_button.setEnabled(not self.running)
        self.retry_button.setEnabled(not self.running and self.selected() is not None)
        self.controls.setEnabled(not self.running)
        self.advanced.setEnabled(not self.running)
        has_output = bool(self.selected() and self.selected()["outputs"])
        self.copy_button.setEnabled(has_output)
        self.open_button.setEnabled(has_output)

    def remove_selected(self):
        rows = {index.row() for index in self.table.selectedIndexes()}
        self.jobs = [job for row, job in enumerate(self.jobs) if row not in rows]
        self.refresh()
        self.preview_selected()

    def retry_selected(self):
        for row in {index.row() for index in self.table.selectedIndexes()}:
            self.jobs[row]["state"] = "En attente"
            self.jobs[row]["error"] = ""
        self.refresh()

    def preview_selected(self):
        job = self.selected()
        text = ""
        if job:
            text = job.get("error", "")
            if job["outputs"]:
                try:
                    text = Path(job["outputs"][0]).read_text(encoding="utf-8")
                except OSError as exc:
                    text = f"Résultat indisponible : {exc}"
        self.preview.setPlainText(text)
        self.update_controls()

    def copy_result(self):
        QApplication.clipboard().setText(self.preview.toPlainText())
        self.status.setText("Texte copié.")

    def reveal_result(self):
        job = self.selected()
        if job and job["outputs"]:
            QProcess.startDetached("/usr/bin/open", ["-R", job["outputs"][0]])

    def start(self):
        self.batch_options = self.options()
        if self.batch_options["cloud"] and not (self.batch_options["api_key"] or os.environ.get("OPENAI_API_KEY")):
            QMessageBox.warning(self, "Clé OpenAI manquante", "Renseignez une clé dans les réglages ou désactivez l’enrichissement OpenAI.")
            return
        for key, value in self.batch_options.items():
            if key not in {"cloud", "api_key"}:
                self.settings.setValue(key, value)
        self.settings.sync()
        self.running = True
        self.cancelled = False
        self.update_controls()
        self.next_job()

    def next_job(self):
        pending = next((j for j in self.jobs if j["state"] == "En attente"), None)
        if self.cancelled or pending is None:
            self.running = False
            self.active = None
            self.progress.setRange(0, 1)
            self.progress.setValue(0 if self.cancelled else 1)
            errors = sum(j["state"] == "Erreur" for j in self.jobs)
            message = "File arrêtée. Les fichiers en attente sont conservés." if self.cancelled else f"Traitement terminé · {errors} erreur(s)."
            self.status.setText(message)
            self.tray.showMessage("Mp4ToTranscript", message)
            QApplication.alert(self)
            self.update_controls()
            if self.closing:
                self.close()
            return
        self.active = pending
        pending["state"] = "Démarrage…"
        pending["error"] = ""
        self.buffer = ""
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.logs = ""
        self.progress.setRange(0, 0)
        process = QProcess(self)
        self.process = process
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PATH", "/opt/homebrew/bin:/usr/local/bin:" + env.value("PATH"))
        for key in ["MallocStackLogging", "MallocStackLoggingNoCompact"]:
            env.remove(key)
        process.setProcessEnvironment(env)
        process.setWorkingDirectory(str(Path(__file__).resolve().parent.parent))
        process.readyReadStandardOutput.connect(self.read_output)
        process.readyReadStandardError.connect(self.read_errors)
        process.finished.connect(self.finished)
        process.errorOccurred.connect(self.process_error)
        request = {"source": pending["source"], "options": self.batch_options,
                   "cache_dir": os.environ.get("MP4_TRANSCRIPT_CACHE_DIR", str(Path.home() / "Library/Caches/Mp4ToTranscript"))}
        def send_request():
            process.write((json.dumps(request) + "\n").encode())
            process.closeWriteChannel()
        process.started.connect(send_request)
        process.start(sys.executable, ["-u", "-m", "mp4_to_transcript.desktop_worker"])
        self.refresh()
        self.table.selectRow(self.jobs.index(pending))

    def read_errors(self):
        self.logs = (self.logs + bytes(self.process.readAllStandardError()).decode("utf-8", "replace"))[-12000:]

    def read_output(self):
        self.buffer += self.decoder.decode(bytes(self.process.readAllStandardOutput()))
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event["event"] == "status":
                self.active["state"] = event["text"]
                self.status.setText(f"{Path(self.active['source']).name} · {event['text']}")
            elif event["event"] == "done":
                self.active["outputs"] = event["outputs"]
                self.active["state"] = "Terminé"
            elif event["event"] == "error":
                self.active["error"] = event["text"]
                self.active["state"] = "Erreur"
        self.refresh()

    def process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.active["error"] = self.process.errorString()
            self.finished(1, QProcess.ExitStatus.CrashExit)

    def finished(self, code, exit_status):
        self.read_output()
        self.read_errors()
        if self.cancelled:
            self.active["state"] = "Annulé"
        elif code != 0 or self.active["state"] != "Terminé":
            self.active["state"] = "Erreur"
            self.active["error"] = self.active["error"] or self.logs or "Le traitement s’est interrompu."
        self.process.deleteLater()
        self.process = None
        self.refresh()
        self.preview_selected()
        QTimer.singleShot(0, self.next_job)

    def cancel(self):
        self.cancelled = True
        self.status.setText("Arrêt en cours…")
        if self.process:
            process = self.process
            process.terminate()
            def force_stop():
                if self.process is process and process.state() != QProcess.ProcessState.NotRunning:
                    process.kill()
            QTimer.singleShot(2500, force_stop)
        self.update_controls()

    def closeEvent(self, event):
        if self.running:
            self.closing = True
            self.cancel()
            event.ignore()
            self.status.setText("Arrêt du traitement avant fermeture…")
        else:
            for key, value in self.options().items():
                if key not in {"cloud", "api_key"}:
                    self.settings.setValue(key, value)
            self.tray.hide()
            event.accept()


STYLE = """
QWidget { font-size: 13px; color: #20231d; }
QMainWindow { background: #fafafa; }
QLabel#heading { font-size: 25px; font-weight: 600; }
QLabel#drop { background: #f3f4f1; border: 1px dashed #869079; border-radius: 10px; color: #444b3b; font-size: 14px; }
QPushButton { min-height: 26px; padding: 3px 12px; border: 1px solid #c7cbc2; border-radius: 6px; background: #fff; }
QPushButton:hover { background: #eef0ea; }
QPushButton:pressed { background: #dde2d4; }
QPushButton:focus { border: 2px solid #536432; }
QPushButton:disabled { color: #767b70; background: #f0f1ee; border-color: #dfe1da; }
QPushButton#primary { background: #3e5026; color: #fff; border-color: #3e5026; font-weight: 600; padding: 5px 22px; }
QPushButton#primary:hover { background: #30421b; }
QPushButton#primary:disabled { background: #e2e5dc; color: #666e5c; border-color: #dfe1da; }
QComboBox, QLineEdit { min-height: 27px; padding: 2px 8px; background: #fff; color: #20231d; }
QComboBox QAbstractItemView { background: #fff; color: #20231d; selection-background-color: #dbe3ce; selection-color: #20231d; }
QTableWidget, QPlainTextEdit { background: #fff; alternate-background-color: #f6f7f4; border: 1px solid #dcded7; border-radius: 5px; selection-background-color: #dbe3ce; selection-color: #20231d; }
QHeaderView::section { background: #f1f3ee; color: #444b3b; border: none; padding: 8px; text-align: left; }
QProgressBar { border: none; background: #e8ebe2; border-radius: 2px; }
QProgressBar::chunk { background: #536432; }
"""


def prepare_macos_platform():
    """Handle files visible to Python but omitted by Qt's directory enumeration.

    Some filesystem providers expose this discrepancy. Materializing our installed
    Cocoa plugin in a private temporary directory lets Qt discover it normally.
    Keep the returned directory alive until QApplication exits.
    """
    if sys.platform != "darwin" or os.environ.get("QT_QPA_PLATFORM"):
        return None
    directory = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)) / "platforms"
    plugin = directory / "libqcocoa.dylib"
    if plugin.is_file() and plugin.name not in QDir(str(directory)).entryList():
        temporary = tempfile.TemporaryDirectory(prefix="mp4-transcript-qt-")
        shutil.copyfile(plugin, Path(temporary.name) / plugin.name)
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = temporary.name
        return temporary
    return None


def main():
    platform_directory = prepare_macos_platform()
    app = QApplication(sys.argv)
    app.setApplicationName("Mp4ToTranscript")
    app.setWindowIcon(QIcon(str(Path(__file__).resolve().parent / "assets" / "app-icon.png")))
    app.setStyle("Fusion")
    palette = app.style().standardPalette()
    for role, color in [(QPalette.ColorRole.Window, "#fafafa"),
                        (QPalette.ColorRole.Base, "#ffffff"),
                        (QPalette.ColorRole.Button, "#ffffff"),
                        (QPalette.ColorRole.Text, "#20231d"),
                        (QPalette.ColorRole.WindowText, "#20231d"),
                        (QPalette.ColorRole.ButtonText, "#20231d")]:
        palette.setColor(role, QColor(color))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor("#606657"))
    app.setPalette(palette)
    app.setStyleSheet(STYLE)
    window = Window()
    window.show()
    if "--smoke-test" in sys.argv:
        def verify_window():
            visible = window.isVisible() and window.windowHandle().isExposed()
            print(f"platform={app.platformName()} window_visible={visible}", flush=True)
            screenshot = os.environ.get("MP4_APP_SMOKE_SCREENSHOT")
            if screenshot:
                window.grab().save(screenshot)
            window.close()
            app.exit(0 if visible else 1)
        QTimer.singleShot(1500, verify_window)
    elif len(sys.argv) > 1:
        window.add_paths(sys.argv[1:])
    result = app.exec()
    if platform_directory:
        platform_directory.cleanup()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
