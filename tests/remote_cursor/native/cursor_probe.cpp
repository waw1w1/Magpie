// Standalone diagnostic. Deliberately does not include or modify Magpie sources.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define UNICODE
#define _UNICODE
#define _WIN32_WINNT 0x0A00
#include <windows.h>
#include <magnification.h>
#include <atomic>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cwchar>
#include <initializer_list>

namespace {
constexpr DWORD Timeout = 1500;
constexpr int EnvironmentUnavailable = 2, ProbeError = 3, CleanupError = 4;
struct Failure { int code; const char* reason; };
void Require(bool ok, const char* name, int code = ProbeError) {
    if (!ok) { std::printf("ERROR api=%s last_error_hint=%lu\n", name, GetLastError()); throw Failure{code, name}; }
}
struct State {
    HWND window = nullptr;
    DWORD ui = 0;
    HCURSOR shapes[2]{};
    int shape = 0;
    unsigned setCursorMessages = 0;
    std::atomic<bool> cancel{false}, attached{false}, queueOk{true}, cleanupOk{true}, countKnown{false};
    HANDLE qRequest{}, qAck{}, qStop{}, qThread{}, oRequest{}, oAck{}, oStop{}, oThread{};
    enum Command { Attach, Read, Hide, Restore, Detach };
    std::atomic<int> command{Read}, down{0}, count{0}, hides{0};
    std::atomic<DWORD> qError{0}, oError{0};
    std::atomic<HCURSOR> qCursor{nullptr}, oCursor{nullptr};
    std::atomic<BOOL> oOk{FALSE};
    std::atomic<DWORD> oFlags{0};
    std::atomic<LONG> oX{0}, oY{0};
} s;
using ShowSystemCursorFn = void (WINAPI*)(BOOL);
ShowSystemCursorFn globalFunction = nullptr;
bool magnificationInitialized = false, globalDirty = false;
const char* backend = "";
HANDLE watchdogDone = nullptr, watchdogThread = nullptr;
[[noreturn]] void EmergencyExit() {
    const char message[] = "RESULT CLEANUP_UNVERIFIED: timeout; terminating only this probe. Global cursor restoration is NOT guaranteed.\n";
    DWORD written = 0; WriteFile(GetStdHandle(STD_ERROR_HANDLE), message, static_cast<DWORD>(sizeof(message) - 1), &written, nullptr);
    TerminateProcess(GetCurrentProcess(), CleanupError);
    std::_Exit(CleanupError); // Last-resort fallback; never claim restoration on this path.
}
DWORD WINAPI Watchdog(void*) {
    if (WaitForSingleObject(watchdogDone, 55000) != WAIT_OBJECT_0) EmergencyExit();
    return 0;
}

LRESULT CALLBACK WindowProc(HWND window, UINT message, WPARAM w, LPARAM l) {
    if (message == WM_SETCURSOR && LOWORD(l) == HTCLIENT) {
        SetCursor(s.shapes[s.shape]); ++s.setCursorMessages; return TRUE;
    }
    if (message == WM_CLOSE) { s.cancel = true; return 0; }
    return DefWindowProcW(window, message, w, l);
}
BOOL WINAPI ConsoleControl(DWORD event) {
    if (event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT) { s.cancel = true; return TRUE; }
    return FALSE;
}
// The UI thread continues dispatching messages during every bounded wait.
bool PumpWait(HANDLE handle, DWORD milliseconds, bool cancelable = true) {
    const ULONGLONG end = GetTickCount64() + milliseconds;
    do {
        MSG message{};
        while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
            if (message.message == WM_QUIT) s.cancel = true;
            TranslateMessage(&message); DispatchMessageW(&message);
        }
        if (cancelable && s.cancel) throw Failure{EnvironmentUnavailable, "cancelled"};
        const ULONGLONG now = GetTickCount64();
        if (now >= end) return !handle;
        const DWORD result = MsgWaitForMultipleObjectsEx(handle ? 1 : 0, handle ? &handle : nullptr,
            static_cast<DWORD>(end - now), QS_ALLINPUT, MWMO_INPUTAVAILABLE);
        if (handle && result == WAIT_OBJECT_0) return true;
        if (result == WAIT_FAILED) return false;
    } while (true);
}
void ProbeAttachedCount() {
    s.down = ShowCursor(FALSE); s.count = ShowCursor(TRUE); s.countKnown = true;
}
void RestoreQueue() {
    while (s.hides > 0) { s.count = ShowCursor(TRUE); --s.hides; }
}
void DetachQueue() {
    RestoreQueue();
    if (s.attached && !AttachThreadInput(GetCurrentThreadId(), s.ui, FALSE)) {
        s.cleanupOk = false; s.qError = GetLastError();
    } else { s.attached = false; s.countKnown = false; }
}
DWORD WINAPI QueueThread(void*) {
    MSG message{}; PeekMessageW(&message, nullptr, 0, 0, PM_NOREMOVE);
    const DWORD self = GetCurrentThreadId();
    const HANDLE events[]{s.qStop, s.qRequest};
    const ULONGLONG deadline = GetTickCount64() + 40000;
    bool stoppedNormally = false;
    while (GetTickCount64() < deadline) {
        DWORD wait = MsgWaitForMultipleObjectsEx(2, events, 500, QS_ALLINPUT, MWMO_INPUTAVAILABLE);
        while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&message); DispatchMessageW(&message);
        }
        if (wait == WAIT_OBJECT_0) { stoppedNormally = true; break; }
        if (wait == WAIT_FAILED) { s.queueOk = false; s.qError = GetLastError(); break; }
        if (wait != WAIT_OBJECT_0 + 1) continue;
        s.qError = 0;
        switch (s.command.load()) {
        case State::Attach: {
            DWORD process = 0;
            const DWORD owner = GetWindowThreadProcessId(s.window, &process);
            if (owner != s.ui || process != GetCurrentProcessId()) {
                s.queueOk = false; s.qError = ERROR_INVALID_WINDOW_HANDLE; break;
            }
            if (!AttachThreadInput(self, owner, TRUE)) { s.queueOk = false; s.qError = GetLastError(); }
            else { s.attached = true; ProbeAttachedCount(); }
            break;
        }
        case State::Hide:
            do { s.count = ShowCursor(FALSE); s.countKnown = true; ++s.hides; }
            while (s.count >= 0 && s.hides < 128);
            if (s.count >= 0) { s.queueOk = false; s.qError = ERROR_INVALID_DATA; }
            break;
        case State::Restore: RestoreQueue(); break;
        case State::Detach: DetachQueue(); break;
        case State::Read: break;
        }
        s.qCursor = GetCursor(); SetEvent(s.qAck);
    }
    if (!stoppedNormally && s.queueOk) { s.queueOk = false; s.qError = ERROR_TIMEOUT; }
    DetachQueue(); SetEvent(s.qAck); return 0;
}
// This thread never attaches, calls ShowCursor/SetCursor, or creates a window.
DWORD WINAPI ObserverThread(void*) {
    const HANDLE events[]{s.oStop, s.oRequest};
    const ULONGLONG deadline = GetTickCount64() + 40000;
    while (GetTickCount64() < deadline) {
        const DWORD wait = WaitForMultipleObjects(2, events, FALSE, 500);
        if (wait == WAIT_OBJECT_0 || wait == WAIT_FAILED) break;
        if (wait != WAIT_OBJECT_0 + 1) continue;
        CURSORINFO info{}; info.cbSize = sizeof(info);
        s.oOk = GetCursorInfo(&info); s.oError = s.oOk ? 0 : GetLastError();
        s.oFlags = info.flags; s.oCursor = info.hCursor; s.oX = info.ptScreenPos.x; s.oY = info.ptScreenPos.y;
        SetEvent(s.oAck);
    }
    return 0;
}
void Queue(State::Command command) {
    s.command = command; Require(SetEvent(s.qRequest), "queue request");
    Require(PumpWait(s.qAck, Timeout), "queue timeout");
    std::printf("QUEUE command=%d attached=%d attach_probe_down=%d count_last_known=%d count_known=%d owned_hides=%d cursor=%p ok=%d error=%lu\n",
        command, int(s.attached.load()), s.down.load(), s.count.load(), int(s.countKnown.load()), s.hides.load(),
        static_cast<void*>(s.qCursor.load()), int(s.queueOk.load()), s.qError.load());
    Require(s.queueOk && s.cleanupOk, "queue operation");
}
void Global(bool show) {
    globalDirty = true; // A failed call also gets a restoration attempt.
    if (globalFunction) {
        globalFunction(show);
        std::printf("GLOBAL backend=%s show=%d status=unreported_void_API\n", backend, int(show));
    } else {
        const BOOL ok = MagShowSystemCursor(show);
        std::printf("GLOBAL backend=%s show=%d ok=%d error=%lu\n", backend, int(show), int(ok), ok ? 0 : GetLastError());
        Require(ok, "MagShowSystemCursor");
    }
    globalDirty = !show;
}
void DesktopGate() {
    DWORD session = 0, bytes = 0;
    wchar_t stationName[256]{}, desktopName[256]{};
    const HWINSTA station = GetProcessWindowStation();
    const HDESK desktop = GetThreadDesktop(GetCurrentThreadId());
    USEROBJECTFLAGS flags{}; BOOL input = FALSE;
    Require(ProcessIdToSessionId(GetCurrentProcessId(), &session), "ProcessIdToSessionId", EnvironmentUnavailable);
    Require(GetUserObjectInformationW(station, UOI_NAME, stationName, sizeof(stationName), &bytes), "station name", EnvironmentUnavailable);
    Require(GetUserObjectInformationW(desktop, UOI_NAME, desktopName, sizeof(desktopName), &bytes), "desktop name", EnvironmentUnavailable);
    Require(GetUserObjectInformationW(station, UOI_FLAGS, &flags, sizeof(flags), &bytes), "station flags", EnvironmentUnavailable);
    Require(GetUserObjectInformationW(desktop, UOI_IO, &input, sizeof(input), &bytes), "desktop UOI_IO", EnvironmentUnavailable);
    const HDESK inputDesktop = OpenInputDesktop(0, FALSE, DESKTOP_READOBJECTS);
    const DWORD openError = inputDesktop ? 0 : GetLastError();
    if (inputDesktop) CloseDesktop(inputDesktop);
    std::printf("ENV session=%lu station=%ls desktop=%ls station_visible=%d input=%d open_input=%d error=%lu\n",
        session, stationName, desktopName, int((flags.dwFlags & WSF_VISIBLE) != 0), int(input), int(inputDesktop != nullptr), openError);
    Require(session != 0 && (flags.dwFlags & WSF_VISIBLE) && input && inputDesktop, "interactive desktop", EnvironmentUnavailable);
}
void TargetGate(POINT expected) {
    DWORD process = 0;
    const HWND target = WindowFromPoint(expected);
    const DWORD owner = GetWindowThreadProcessId(target, &process);
    POINT actual{};
    Require(GetCursorPos(&actual), "GetCursorPos", EnvironmentUnavailable);
    Require(IsWindowVisible(s.window) && GetForegroundWindow() == s.window && target == s.window &&
        owner == s.ui && process == GetCurrentProcessId() && actual.x == expected.x && actual.y == expected.y,
        "foreground / pointer target / owner", EnvironmentUnavailable);
    BOOL input = FALSE; DWORD bytes = 0;
    Require(GetUserObjectInformationW(GetThreadDesktop(s.ui), UOI_IO, &input, sizeof(input), &bytes) && input,
        "desktop stopped receiving input", EnvironmentUnavailable);
}
bool Snapshot(const char* stage, int shape, POINT point) {
    TargetGate(point);
    Require(SetEvent(s.oRequest), "observer request");
    Require(PumpWait(s.oAck, Timeout), "observer timeout");
    TargetGate(point);
    std::printf("OBSERVER ok=%d error=%lu\n", int(s.oOk.load()), s.oError.load());
    Require(s.oOk, "GetCursorInfo observer");
    const bool handleMatches = s.oCursor == s.shapes[shape];
    const bool positionMatches = s.oX == point.x && s.oY == point.y;
    std::printf("SAMPLE stage=%s shape=%c setcursor_messages=%u expected_handle=%p expected_xy=%ld,%ld "
        "flags=%lu handle=%p xy=%ld,%ld handle_match=%d position_match=%d observer_attachment=none error=%lu\n",
        stage, 'A' + shape, s.setCursorMessages, static_cast<void*>(s.shapes[shape]), point.x, point.y,
        s.oFlags.load(), static_cast<void*>(s.oCursor.load()), s.oX.load(), s.oY.load(), int(handleMatches),
        int(positionMatches), s.oError.load());
    const bool follows = handleMatches && positionMatches && s.oFlags == CURSOR_SHOWING;
    Queue(State::Read); // GetCursor only: no ShowCursor measurement pulses in any stage.
    std::printf("QUEUE_SHAPE stage=%s shape=%c match=%d\n", stage, 'A' + shape, int(s.qCursor == s.shapes[shape]));
    return follows;
}
void Transition(const char* stage, int shape, POINT point, bool strict) {
    s.shape = shape;
    const unsigned before = s.setCursorMessages;
    Require(SetCursorPos(point.x, point.y), "SetCursorPos", EnvironmentUnavailable);
    // Require delivery through the input desktop, not a manually sent WM_SETCURSOR.
    const ULONGLONG deadline = GetTickCount64() + Timeout;
    do { PumpWait(nullptr, 20); } while (strict && s.setCursorMessages == before && GetTickCount64() < deadline);
    if (strict) Require(s.setCursorMessages > before, "WM_SETCURSOR not delivered", EnvironmentUnavailable);
    // Hidden stages explicitly change the owner's cursor, even if Windows suppresses WM_SETCURSOR.
    const HCURSOR previous = SetCursor(s.shapes[shape]);
    std::printf("OWNER stage=%s shape=%c prior=%p current=%p move=ok\n", stage, 'A' + shape,
        static_cast<void*>(previous), static_cast<void*>(GetCursor()));
    for (int sample = 0; sample != 2; ++sample) {
        PumpWait(nullptr, 50);
        const bool follows = Snapshot(stage, shape, point);
        if (strict) Require(follows, "baseline observer does not follow A/B", EnvironmentUnavailable);
    }
}
} // namespace

int wmain(int argc, wchar_t** argv) {
    if (argc != 3 || wcscmp(argv[1], L"--initial-global-visible") ||
        (wcscmp(argv[2], L"--backend=user32") && wcscmp(argv[2], L"--backend=magnification"))) {
        std::puts("ENVIRONMENT_UNAVAILABLE: requires --initial-global-visible --backend=user32|magnification; "
            "isolated interactive desktop only. No global-hide getter exists in the public Magnification API.");
        return EnvironmentUnavailable;
    }
    setvbuf(stdout, nullptr, _IONBF, 0);
    int result = 0, originalCount = 0; bool countSaved = false, positionSaved = false;
    POINT originalPosition{}; HWND originalForeground = GetForegroundWindow();
    s.ui = GetCurrentThreadId(); backend = wcscmp(argv[2], L"--backend=user32") ? "magnification" : "user32";
    try {
        watchdogDone = CreateEventW(nullptr, TRUE, FALSE, nullptr); Require(watchdogDone != nullptr, "watchdog event");
        watchdogThread = CreateThread(nullptr, 0, Watchdog, nullptr, 0, nullptr); Require(watchdogThread != nullptr, "watchdog thread");
        DesktopGate();
        if (backend[0] == 'u') {
            globalFunction = std::bit_cast<ShowSystemCursorFn>(GetProcAddress(GetModuleHandleW(L"user32.dll"), "ShowSystemCursor"));
            Require(globalFunction != nullptr, "ShowSystemCursor unavailable", EnvironmentUnavailable);
        } else {
            Require(MagInitialize(), "MagInitialize", EnvironmentUnavailable); magnificationInitialized = true;
        }
        Require(GetCursorPos(&originalPosition), "save pointer position", EnvironmentUnavailable); positionSaved = true;
        ShowCursor(FALSE); originalCount = ShowCursor(TRUE); countSaved = true;
        Require(originalCount >= 0 && originalCount < 64, "probe requires initial UI count in [0,63]", EnvironmentUnavailable);
        s.shapes[0] = LoadCursorW(nullptr, IDC_ARROW); s.shapes[1] = LoadCursorW(nullptr, IDC_CROSS);
        Require(s.shapes[0] && s.shapes[1] && s.shapes[0] != s.shapes[1], "load distinct shared cursors");
        WNDCLASSW wc{}; wc.lpfnWndProc = WindowProc; wc.hInstance = GetModuleHandleW(nullptr);
        wc.lpszClassName = L"MagpieCursorProbe"; wc.hCursor = s.shapes[0]; wc.hbrBackground = reinterpret_cast<HBRUSH>(static_cast<INT_PTR>(COLOR_WINDOW + 1));
        Require(RegisterClassW(&wc), "RegisterClassW");
        s.window = CreateWindowExW(0, wc.lpszClassName, L"Cursor metadata probe (close to cancel)", WS_OVERLAPPEDWINDOW,
            120, 120, 520, 260, nullptr, nullptr, wc.hInstance, nullptr);
        Require(s.window != nullptr, "CreateWindowExW"); ShowWindow(s.window, SW_SHOW); UpdateWindow(s.window);
        const BOOL foreground = SetForegroundWindow(s.window);
        std::printf("WINDOW own_hwnd=%p ui_tid=%lu foreground_request=%d initial_count=%d backend=%s\n",
            static_cast<void*>(s.window), s.ui, int(foreground), originalCount, backend);
        for (HANDLE* event : {&s.qRequest, &s.qAck, &s.qStop, &s.oRequest, &s.oAck, &s.oStop}) {
            *event = CreateEventW(nullptr, FALSE, FALSE, nullptr); Require(*event != nullptr, "CreateEventW");
        }
        s.qThread = CreateThread(nullptr, 0, QueueThread, nullptr, 0, nullptr);
        s.oThread = CreateThread(nullptr, 0, ObserverThread, nullptr, 0, nullptr);
        Require(s.qThread && s.oThread, "CreateThread");
        std::printf("THREADS worker_tid=%lu observer_tid=%lu observer_attachment=never_same_process\n",
            GetThreadId(s.qThread), GetThreadId(s.oThread));
        SetConsoleCtrlHandler(ConsoleControl, TRUE);
        POINT points[2]{{180, 90}, {205, 105}};
        Require(ClientToScreen(s.window, &points[0]) && ClientToScreen(s.window, &points[1]), "ClientToScreen");
        // Start at B so the baseline A transition necessarily moves the pointer.
        Require(SetCursorPos(points[1].x, points[1].y), "initial pointer positioning", EnvironmentUnavailable); PumpWait(nullptr, 100);
        TargetGate(points[1]);
        CURSORINFO initial{}; initial.cbSize = sizeof(initial);
        Require(GetCursorInfo(&initial), "initial cursor info", EnvironmentUnavailable);
        std::printf("BASELINE_INPUT flags=%lu handle=%p\n", initial.flags, static_cast<void*>(initial.hCursor));
        if (initial.flags & CURSOR_SUPPRESSED) {
            // SetCursorPos does not establish mouse input on every hosted desktop.
            // One ordinary move inside our own client area may leave touch/pen mode.
            // Do not alter system settings or weaken the observer's baseline gate.
            INPUT input{}; input.type = INPUT_MOUSE;
            input.mi.dx = 1; input.mi.dwFlags = MOUSEEVENTF_MOVE;
            const UINT sent = SendInput(1, &input, sizeof(input));
            const DWORD error = sent == 1 ? 0 : GetLastError();
            std::printf("BASELINE_INPUT mouse_move_sent=%u error=%lu buttons=none\n", sent, error);
            Require(sent == 1, "baseline mouse move", EnvironmentUnavailable);
            PumpWait(nullptr, 100);
            CURSORINFO afterInput{}; afterInput.cbSize = sizeof(afterInput);
            Require(GetCursorInfo(&afterInput), "cursor info after mouse move", EnvironmentUnavailable);
            std::printf("BASELINE_INPUT after_mouse_move_flags=%lu handle=%p xy=%ld,%ld\n",
                afterInput.flags, static_cast<void*>(afterInput.hCursor), afterInput.ptScreenPos.x, afterInput.ptScreenPos.y);
        }
        Transition("baseline_unattached", 0, points[0], true); Transition("baseline_unattached", 1, points[1], true);
        Queue(State::Attach);
        Transition("baseline_attached", 0, points[0], true); Transition("baseline_attached", 1, points[1], true);
        const char* stages[]{"queue_only", "global_only", "global_then_queue_restore_GQ", "queue_then_global_restore_GQ",
            "global_then_queue_restore_QG", "queue_then_global_restore_QG"};
        for (int stage = 0; stage != 6; ++stage) {
            Transition("seed_visible", 0, points[0], true);
            if (stage == 1 || stage == 2 || stage == 4) Global(false);
            if (stage != 1) Queue(State::Hide);
            if (stage == 3 || stage == 5) Global(false);
            std::printf("STILL stage=%s before_owner_changes=1\n", stages[stage]);
            Snapshot(stages[stage], 0, points[0]);
            Transition(stages[stage], 0, points[0], false); Transition(stages[stage], 1, points[1], false);
            if (stage >= 4) {
                Queue(State::Restore); PumpWait(nullptr, 80);
                Snapshot("after_queue_restore_before_global", 1, points[1]);
            }
            if (stage != 0) { Global(true); PumpWait(nullptr, 80); Snapshot("after_global_restore_before_move", 1, points[1]); }
            if (stage < 4) Queue(State::Restore);
            PumpWait(nullptr, 80);
            Snapshot("restored_before_move", 1, points[1]); // Catch-up without movement is an observation, not an assertion.
            Transition("restored_reactivated", 0, points[0], true);
            Transition("restored_reactivated", 1, points[1], true);
        }
        Snapshot("before_detach", 1, points[1]); Queue(State::Detach);
        Snapshot("after_detach", 1, points[1]);
    } catch (const Failure& failure) {
        result = failure.code; std::printf("STOP reason=%s code=%d\n", failure.reason, failure.code);
    } catch (...) { result = ProbeError; std::puts("STOP unexpected C++ exception"); }
    // Keep the window and pump alive until the attached worker has undone only its own decrements.
    if (s.qStop) SetEvent(s.qStop);
    const bool workerStopped = !s.qThread || PumpWait(s.qThread, 2000, false);
    if (!workerStopped) EmergencyExit(); // No potentially synchronizing GUI cleanup after an attached worker timeout.
    if (s.oStop) SetEvent(s.oStop);
    const bool observerStopped = !s.oThread || PumpWait(s.oThread, 2000, false);
    if (!s.queueOk && result == 0) result = ProbeError;
    if (!workerStopped || !observerStopped || !s.cleanupOk || s.attached || s.hides) result = CleanupError;
    try { if (globalDirty) Global(true); } catch (...) { result = CleanupError; }
    if (countSaved && workerStopped) {
        ShowCursor(FALSE); int count = ShowCursor(TRUE);
        std::printf("CLEANUP ui_count_before=%d expected=%d\n", count, originalCount);
        if (count != originalCount) result = CleanupError; // Never hide a failed restoration check.
        for (int retry = 0; count != originalCount && retry != 128; ++retry) count = ShowCursor(count < originalCount);
        std::printf("CLEANUP ui_count_after=%d restored=%d\n", count, int(count == originalCount));
    }
    if (positionSaved) {
        const BOOL moved = SetCursorPos(originalPosition.x, originalPosition.y); POINT actual{};
        const BOOL read = GetCursorPos(&actual);
        const bool restored = moved && read && actual.x == originalPosition.x && actual.y == originalPosition.y;
        std::printf("CLEANUP pointer_restored=%d\n", int(restored)); if (!restored) result = CleanupError;
    }
    if (s.window) {
        const bool ownForeground = GetForegroundWindow() == s.window;
        DestroyWindow(s.window);
        if (ownForeground && originalForeground && IsWindow(originalForeground)) SetForegroundWindow(originalForeground);
    }
    if (magnificationInitialized && !MagUninitialize()) result = CleanupError;
    std::printf("CLEANUP worker_stopped=%d observer_stopped=%d attached=%d hides=%d error=%lu\n",
        int(workerStopped), int(observerStopped), int(s.attached.load()), s.hides.load(), s.qError.load());
    std::printf("RESULT %s code=%d capture_test=not_performed remote_receiver_test=not_performed\n",
        result == 0 ? "OBSERVATION_COMPLETE" : result == EnvironmentUnavailable ? "ENVIRONMENT_UNAVAILABLE" :
        result == CleanupError ? "CLEANUP_FAILED_OR_UNVERIFIED" : "PROBE_ERROR", result);
    // A stuck Windows API cannot be repaired safely; the nonzero result never claims cleanup succeeded.
    if (!observerStopped) EmergencyExit();
    if (watchdogDone) SetEvent(watchdogDone);
    if (watchdogThread && WaitForSingleObject(watchdogThread, 1000) != WAIT_OBJECT_0) EmergencyExit();
    if (watchdogThread) CloseHandle(watchdogThread);
    if (watchdogDone) CloseHandle(watchdogDone);
    for (HANDLE event : {s.qRequest, s.qAck, s.qStop, s.qThread, s.oRequest, s.oAck, s.oStop, s.oThread}) if (event) CloseHandle(event);
    return result;
}
