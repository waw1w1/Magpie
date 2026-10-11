"""Build with installed VS; visible cursor probing requires explicit isolated-desktop opt-in."""
from pathlib import Path
import os
import argparse
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-isolated-desktop", action="store_true",
                        help="Opt in to visible UI/cursor changes; assert initial global cursor is visible and no competing cursor manager exists")
    args = parser.parse_args(argv)
    if os.name != "nt":
        print("BUILD_UNAVAILABLE: Windows + installed Visual Studio C++ tools are required.")
        return 2
    print(f"WINDOWS_VERSION {sys.getwindowsversion()}", flush=True)
    root = Path(__file__).resolve().parent
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / (
        "Microsoft Visual Studio/Installer/vswhere.exe")
    if not vswhere.is_file():
        print("BUILD_UNAVAILABLE: installed vswhere.exe was not found; nothing will be installed.")
        return 2
    found = subprocess.run(
        [str(vswhere), "-latest", "-prerelease", "-products", "*", "-requires",
         "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath", "-utf8"],
        capture_output=True, text=True, encoding="utf-8", check=True, timeout=30)
    installations = found.stdout.strip().splitlines()
    if not installations:
        print("BUILD_UNAVAILABLE: no installed Visual Studio x64 C++ toolchain was found.")
        return 2
    devcmd = Path(installations[0]) / "Common7/Tools/VsDevCmd.bat"
    if not devcmd.is_file():
        print("BUILD_UNAVAILABLE: the selected installation has no VsDevCmd.bat.")
        return 2
    out = root / "out"
    out.mkdir(exist_ok=True)
    # A local batch file avoids cmd /c's outer-quote rules. All outputs stay beside the prototype.
    source = root / "cursor_probe.cpp"
    for path in (devcmd, source, out):
        if any(char in str(path) for char in '\r\n%"'):
            raise ValueError("Unsupported batch-file path character")
    batch = out / "build.cmd"
    batch.write_text(
        '@echo off\n'
        f'call "{devcmd}" -arch=x64 -host_arch=x64\n'
        'if errorlevel 1 exit /b %errorlevel%\n'
        f'cl.exe /nologo /std:c++20 /EHsc /W4 /WX /permissive- /utf-8 /O2 /MT "{source}" '
        '/Fe:cursor_probe.exe /Fo:cursor_probe.obj /link user32.lib magnification.lib\n'
        'exit /b %errorlevel%\n', encoding="utf-8")
    completed = subprocess.run(["cmd.exe", "/d", "/c", str(batch)], cwd=out, timeout=180)
    if completed.returncode == 0:
        print(f"BUILD_COMPLETE: {out / 'cursor_probe.exe'}")
    if completed.returncode != 0:
        return completed.returncode
    if not args.run_isolated_desktop:
        print("NOT_RUN: building does not start visible UI or alter cursor state.")
        return 0
    result = 0
    for backend in ("user32", "magnification"):
        print(f"RUN backend={backend} isolated_desktop_opt_in=true initial_global_visible_asserted=true", flush=True)
        try:
            run = subprocess.run([str(out / "cursor_probe.exe"), "--initial-global-visible", f"--backend={backend}"], timeout=75)
        except subprocess.TimeoutExpired:
            print("CLEANUP_FAILED_OR_UNVERIFIED: native probe timed out; do not continue on this desktop.", flush=True)
            return 4
        print(f"BACKEND_RESULT backend={backend} exit={run.returncode}", flush=True)
        if run.returncode not in (0, 2, 3):
            print("CLEANUP_FAILED_OR_UNVERIFIED: refusing to start the other backend.", flush=True)
            return 4
        result = max(result, run.returncode)
    return result


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"BUILD_ERROR: {error}")
        sys.exit(3)
