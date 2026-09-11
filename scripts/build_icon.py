"""Render the vector source into the PNG and ICNS used by the desktop app."""
import subprocess
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtSvg import QSvgRenderer


def build_icon():
    assets = Path(__file__).resolve().parents[1] / "mp4_to_transcript" / "assets"
    renderer = QSvgRenderer(str(assets / "app-icon.svg"))
    if not renderer.isValid():
        raise RuntimeError("Invalid icon SVG")

    def render(size, destination):
        image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        renderer.render(painter)
        painter.end()
        if not image.save(str(destination)):
            raise RuntimeError(f"Cannot write {destination}")

    render(1024, assets / "app-icon.png")
    with tempfile.TemporaryDirectory() as directory:
        iconset = Path(directory) / "AppIcon.iconset"
        iconset.mkdir()
        for size in (16, 32, 128, 256, 512):
            render(size, iconset / f"icon_{size}x{size}.png")
            render(size * 2, iconset / f"icon_{size}x{size}@2x.png")
        subprocess.run(["/usr/bin/iconutil", "-c", "icns", str(iconset),
                        "-o", str(assets / "AppIcon.icns")], check=True)


if __name__ == "__main__":
    build_icon()
