# Native cursor metadata probe

This standalone Win32 experiment measures cursor metadata, not captured frames
or remote-client rendering. It does not include the Magpie application. A completed
matrix is **not evidence that Steam Link / OBS double cursors are fixed**.

## Explicit opt-in only

Use an isolated, disposable Windows desktop with Visual Studio C++ tools and the
Windows SDK already installed. The global cursor must initially be visible, with
no competing cursor manager. The probe temporarily changes its own foreground
window, pointer position and cursor visibility. **Do not run on an active user's
desktop. A crash or hard timeout may prevent cursor restoration.**

From this directory:

```text
python build.py
python build.py --run-isolated-desktop
```

The first command only builds `out/cursor_probe.exe`, using installed `vswhere`
and `VsDevCmd` with C++20 and `/W4 /WX`. It prints the actual Windows build.
The second explicitly runs both backends in separate processes. An unknown,
crash, cleanup-failure or hard-timeout result prevents the next run.

Alternatively, after building, select one backend:

```text
out\cursor_probe.exe --initial-global-visible --backend=user32
out\cursor_probe.exe --initial-global-visible --backend=magnification
```

The `build` workflow runs the two backends on separate Windows runners only when
`workflow_dispatch` explicitly enables `remote_cursor_probe` (default: false).
Ordinary push/PR builds do not execute this UI experiment. No software is installed
by the launcher; no RDP, autologon, desktop assignment or security settings change.

## What is measured

The UI thread owns a visible test window and pumps messages. A windowless worker
attaches to that thread. A separate observer thread never attaches or changes any
cursor state. It is **in the same process**, not an independent capture process.

Before observations, the probe requires a visible window station, input desktop,
matching foreground/pointer/window ownership, real `WM_SETCURSOR` delivery, and
two matching global snapshots for each of two shared cursor shapes and positions.
A runner label or successful `OpenInputDesktop` alone does not satisfy this gate.
Initial UI display counts outside `[0,63]` conservatively fail a probe prerequisite;
that does not prove the desktop itself is noninteractive.

If Windows initially reports `CURSOR_SUPPRESSED`, the probe tries one ordinary
relative `SendInput` mouse move (`dx=1`) inside its verified own client area, without clicks,
before the baseline. This may switch away from touch/pen input; it is not guaranteed.
The same strict observer baseline must still pass, or the experiment is unavailable.
No synthetic input is added during the hiding-order observations.

For each global backend, the matrix covers queue hiding alone, global hiding alone,
global→queue and queue→global hiding, both restoration orders, A→B shape/position
changes while hidden, and still observations before/after explicit detach.
Before the already-attached matrix, a fresh worker follows Magpie's initial handoff order:
global hide → attach → balanced queue-count probe → queue hide. It then balances
the queue and detaches before restoring global visibility, as at shutdown.
Read-only still samples separate attachment, the balanced count probe, and hiding
so any metadata transition can be attributed to the appropriate operation.

Each case seeds visible A and samples immediately after hiding, **before** changing
the owner's cursor or position. It also samples between restoration operations and
before moving again. Catch-up behavior is recorded, not asserted; the next case
must establish a fresh visible baseline.

`SAMPLE` logs global flags/handle/position and expected values. `QUEUE` logs the
worker's shape, owned hide balance and **last-known** display count. Explicit
`ProbeCount` commands use balanced ShowCursor calls; normal
snapshots call GetCursor only, so measurement does not pulse cursor visibility.
The user32 backend uses Magpie's undocumented `ShowSystemCursor` signature and
explicitly logs that the void API cannot report success. The Magnification backend
records its Boolean result. No fallback silently changes the selected backend.

Exit results:

- 0: `OBSERVATION_COMPLETE`; matrix and cleanup completed, not a capture test pass.
- 2: `ENVIRONMENT_UNAVAILABLE` or another unmet prerequisite; no supported result.
- 3: `PROBE_ERROR`, including an API failure or operation timeout.
- 4: `CLEANUP_FAILED_OR_UNVERIFIED`; restoration must not be assumed.

Normal cleanup balances worker-owned hides, detaches, restores outstanding global
hiding, checks the UI count, restores the original pointer position, and closes its
window. A count mismatch remains an error even if corrected on the probe's own UI
thread. There is no public Magnification getter for the prior global-hide switch,
so the initial-visible/no-competing-manager precondition cannot be omitted.

An unresponsive attached worker prevents further GUI cleanup. A separate 55-second
watchdog terminates only this probe; the optional Python runner has a 75-second
outer timeout. These paths explicitly do **not** guarantee global restoration or
absence of attached-queue deadlocks. Closing the window or Ctrl+C requests graceful
cleanup; forced termination does not.

## Portable launcher tests

```text
python -B -m unittest discover -s . -p test_build.py -v
```

These mock Windows subprocess calls to check build-only behavior, explicit opt-in,
and error propagation. They do not compile against the Windows SDK or execute Win32.

## API references

- [GetCursorInfo](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getcursorinfo)
- [ShowCursor](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-showcursor)
- [AttachThreadInput](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-attachthreadinput)
- [GetUserObjectInformation](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getuserobjectinformationw)
- [MagShowSystemCursor](https://learn.microsoft.com/en-us/windows/win32/api/magnification/nf-magnification-magshowsystemcursor)
- [SendInput](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendinput)
