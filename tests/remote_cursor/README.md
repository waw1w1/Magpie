# RemoteCursor state-machine regression tests

Run on Linux with Python 3 and a C++20 compiler:

```sh
python3 tests/remote_cursor/run.py
```

The runner copies the **actual, unchanged** `RemoteCursor.cpp` and
`RemoteCursor.h` into a temporary directory. Lightweight `pch.h` and `Logger.h`
stand in for Win32, wil, and logging; the cursor algorithm is not reimplemented.
No production hooks, Windows SDK, test framework, or generated checkout files
are needed. `CXX` and `CXXFLAGS` are supported. To check a historical revision,
export its two source files into a temporary directory and pass `--source-dir`.

A real worker thread is gated at each fake Win32 wait. Each test advances one
wake and waits for the next wait (or thread exit), without timing sleeps. Queue
counts, shapes, failures, and source-side `ShowCursor` deltas are controlled only
while that worker is parked. Assertions cover:

- Pending requests suppress software rendering; completed requests preserve the
  source's logical visibility and cursor shape.
- Owner switches and cancellation invalidate old snapshots, including switching
  back before the worker processes an intermediate owner.
- Restoring balances only worker-owned hides, including initially negative and
  positive counts and source-side visibility changes.
- Attach/event/wait failures preserve fallback input; failed detach cannot be
  reported as a successful detach or permit another attachment.
- `CURSOR_SUPPRESSED` is respected and input-message wakes are pumped.

These are deterministic state-machine tests, **not Windows integration tests**.
The fakes model shared counts and queue ownership; they cannot prove Windows'
actual `AttachThreadInput` behavior, focus/capture side effects, source-thread
exit races, hardware cursor capture, or what a remote client displays. Those
still require a Windows build and interactive remote-capture verification.

An explicitly opted-in [native metadata probe](native/README.md) can additionally
measure the order of queue/global hiding on an isolated Windows input desktop.
It does not capture frames or connect to a remote client.
