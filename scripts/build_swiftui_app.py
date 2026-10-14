"""Build the SwiftUI app: Swift interface + PyInstaller worker bundle in Contents/Helpers, signed.

Ad hoc signature by default (`--identity -`). With a Developer ID later, pass
`--identity "Developer ID Application: …"`: the hardened runtime is then enabled.

A copy left inside iCloud Drive (~/Documents) gets Finder attributes back and fails
`codesign --verify --strict`; `--install` copies the signed app to /Applications instead.

`--release` builds the shareable app: final name and bundle identifier (it replaces the
Qt app of the same name), plus a versioned .zip to send to colleagues.
"""
import argparse
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "Mp4ToTranscript"
WORKER = "mp4-transcript-worker"
WORKER_APP = "Mp4ToTranscriptWorker.app"
MACHO_MAGIC = (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xfe\xed\xfa\xcf")
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
AUDIO_VIDEO = ["aac", "flac", "m4a", "m4v", "mkv", "mov", "mp3", "mp4", "mpeg", "mpga",
               "ogg", "opus", "wav", "webm", "wma"]


def build_worker(work: Path) -> Path:
    env = dict(os.environ, PYINSTALLER_CONFIG_DIR=str(work / "pyinstaller-cache"))
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--log-level", "WARN",
                    "--distpath", str(work / "worker-dist"), "--workpath", str(work / "worker-work"),
                    str(ROOT / "scripts/Worker.spec")], cwd=ROOT, env=env, check=True)
    return work / "worker-dist" / WORKER_APP


def build_interface() -> Path:
    package = ROOT / "macos"
    subprocess.run(["swift", "build", "-c", "release", "--arch", "arm64"], cwd=package, check=True)
    binary = subprocess.run(["swift", "build", "-c", "release", "--arch", "arm64", "--show-bin-path"],
                            cwd=package, check=True, capture_output=True, text=True).stdout.strip()
    return Path(binary) / NAME


def install_icon(resources: Path) -> bool:
    """Compile macos/AppIcon.icon (Icon Composer) with actool, from Xcode.

    Without Xcode, fall back to the flat macos/Resources/AppIcon.icns.
    Returns True when the Liquid Glass icon (Assets.car) was installed.
    """
    try:
        subprocess.run(["xcrun", "--find", "actool"], check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        shutil.copy2(ROOT / "macos/Resources/AppIcon.icns", resources / "AppIcon.icns")
        print("actool absent (Xcode non installé) : icône .icns de secours")
        return False
    with tempfile.TemporaryDirectory() as partial:
        subprocess.run(["xcrun", "actool", str(ROOT / "macos/AppIcon.icon"), "--compile", str(resources),
                        "--app-icon", "AppIcon", "--platform", "macosx",
                        "--minimum-deployment-target", "26.0", "--target-device", "mac",
                        "--output-partial-info-plist", str(Path(partial) / "icon.plist")],
                       check=True, capture_output=True)
    return True


def info_plist(bundle_id: str, version: str) -> dict:
    return {
        "CFBundleName": NAME, "CFBundleDisplayName": NAME, "CFBundleExecutable": NAME,
        "CFBundleIdentifier": bundle_id, "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": version, "CFBundleVersion": version,
        "CFBundleIconFile": "AppIcon", "CFBundleIconName": "AppIcon", "CFBundleDevelopmentRegion": "fr",
        "LSMinimumSystemVersion": "26.0", "LSArchitecturePriority": ["arm64"],
        "LSApplicationCategoryType": "public.app-category.productivity",
        "NSHighResolutionCapable": True,
        "NSHumanReadableCopyright": "© shigaepouyen",
        "CFBundleDocumentTypes": [{
            "CFBundleTypeName": "Audio et vidéo", "CFBundleTypeRole": "Viewer",
            "LSHandlerRank": "Alternate",
            "LSItemContentTypes": ["public.audio", "public.movie"],
            "CFBundleTypeExtensions": AUDIO_VIDEO}],
    }


def is_macho(path: Path) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    with path.open("rb") as stream:
        return stream.read(4) in MACHO_MAGIC


def sign(app: Path, identity: str):
    """Sign inside-out: every nested Mach-O, then the worker bundle, then the app."""
    options = [] if identity == "-" else ["--options", "runtime", "--timestamp"]
    codesign = ["codesign", "--force", "--sign", identity, *options]
    subprocess.run(["xattr", "-cr", str(app)], check=True)  # codesign rejects Finder "detritus"
    helper = app / "Contents/Helpers" / WORKER_APP
    nested = sorted((p for p in helper.rglob("*") if is_macho(p) and p.name != WORKER),
                    key=lambda p: len(p.parts), reverse=True)
    for start in range(0, len(nested), 200):
        subprocess.run([*codesign, *map(str, nested[start:start + 200])], check=True,
                       capture_output=True)
    subprocess.run([*codesign, str(helper)], check=True)
    subprocess.run([*codesign, str(app)], check=True)
    subprocess.run(["codesign", "--verify", "--deep", "--strict", "--verbose=2", str(app)], check=True)
    return len(nested) + 2


def assemble(destination: Path, work: Path, bundle_id: str, version: str,
             identity: str, skip_worker: bool, install: str | None, archive: bool = False) -> Path:
    worker = work / "worker-dist" / WORKER_APP if skip_worker else build_worker(work)
    if not worker.is_dir():
        raise SystemExit(f"Worker introuvable : {worker}")
    interface = build_interface()
    # Assemble and sign outside iCloud Drive: File Provider tags bundle folders with
    # com.apple.FinderInfo, which codesign rejects.
    staging = Path(tempfile.mkdtemp(prefix="mp4-swiftui-"))
    app = staging / f"{NAME}.app"
    (app / "Contents/MacOS").mkdir(parents=True)
    (app / "Contents/Resources").mkdir()
    shutil.copy2(interface, app / "Contents/MacOS" / NAME)
    shutil.copytree(worker, app / "Contents/Helpers" / WORKER_APP, symlinks=True)
    install_icon(app / "Contents/Resources")
    with (app / "Contents/Info.plist").open("wb") as stream:
        plistlib.dump(info_plist(bundle_id, version), stream)
    (app / "Contents/PkgInfo").write_text("APPL????")
    signed = sign(app, identity)
    print(f"{signed} binaires signés ({'ad hoc' if identity == '-' else identity})")
    final = destination / app.name
    shutil.rmtree(final, ignore_errors=True)
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ditto", "--noextattr", str(app), str(final)], check=True)
    if archive:
        zip_path = destination / f"{NAME}-{version}.zip"
        zip_path.unlink(missing_ok=True)
        # ditto keeps the signature and symlinks intact, unlike zip.
        subprocess.run(["ditto", "-c", "-k", "--keepParent", "--norsrc", "--noextattr",
                        str(app), str(zip_path)], check=True)
        print(f"Archive : {zip_path}")
    if install:
        installed = Path("/Applications") / install
        shutil.rmtree(installed, ignore_errors=True)
        subprocess.run(["ditto", "--noextattr", str(app), str(installed)], check=True)
        subprocess.run(["codesign", "--verify", "--deep", "--strict", str(installed)], check=True)
        print(f"Installée et vérifiée : {installed}")
    shutil.rmtree(staging)
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "build/swiftui")
    parser.add_argument("--work", type=Path, default=ROOT / "build/swiftui")
    # Distinct identifier while the Qt app (same name) stays installed alongside.
    parser.add_argument("--bundle-id", default="io.github.shigaepouyen.mp4totranscript.swiftui")
    parser.add_argument("--version", default=VERSION, help="Default: pyproject.toml")
    parser.add_argument("--release", action="store_true",
                        help="Final bundle identifier and a versioned .zip")
    parser.add_argument("--identity", default="-")
    parser.add_argument("--install", metavar="NOM.app",
                        help="Also copy the signed app to /Applications under this name")
    parser.add_argument("--skip-worker", action="store_true",
                        help="Reuse the worker already built in --work")
    args = parser.parse_args()
    bundle_id = "io.github.shigaepouyen.mp4totranscript" if args.release else args.bundle_id
    print(assemble(args.destination.resolve(), args.work.resolve(), bundle_id,
                   args.version, args.identity, args.skip_worker, args.install, args.release))
