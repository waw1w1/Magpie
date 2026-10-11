#!/usr/bin/env python3
"""Compile unchanged production RemoteCursor sources against tiny Linux fakes."""

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path,
                        help="directory containing alternate RemoteCursor.cpp/.h")
    args = parser.parse_args()
    tests = Path(__file__).resolve().parent
    source = args.source_dir or tests.parents[1] / "src" / "Magpie.Core"
    compiler = os.environ.get("CXX") or shutil.which("g++-14") or shutil.which("g++")
    if not compiler:
        parser.error("a C++20 compiler is required (set CXX)")

    with tempfile.TemporaryDirectory(prefix="magpie-remote-cursor-") as directory:
        build = Path(directory)
        for name in ("RemoteCursor.cpp", "RemoteCursor.h"):
            shutil.copyfile(source / name, build / name)
        for path in (tests / "fakes").iterdir():
            shutil.copyfile(path, build / path.name)
        shutil.copyfile(tests / "test_remote_cursor.cpp", build / "test.cpp")
        command = shlex.split(compiler) + [
            "-std=c++20", "-pthread", "-Wall", "-Wextra", "-Werror",
            "-Wno-missing-field-initializers", "-O1", "-g",
            *shlex.split(os.environ.get("CXXFLAGS", "")),
            "RemoteCursor.cpp", "test.cpp", "-o", "remote_cursor_tests",
        ]
        print("+", shlex.join(command), flush=True)
        subprocess.run(command, cwd=build, check=True, timeout=60)
        return subprocess.run([str(build / "remote_cursor_tests")], timeout=30).returncode


if __name__ == "__main__":
    raise SystemExit(main())
