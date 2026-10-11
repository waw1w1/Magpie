#pragma once

// Small Win32/wil surface for the actual RemoteCursor translation unit.
// Queue state may be inspected by tests only while the worker is parked/exited.
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using DWORD = std::uint32_t;
using BOOL = int;
using HANDLE = void*;
using HCURSOR = void*;
struct POINT { long x; long y; };
struct CURSORINFO { DWORD cbSize; DWORD flags; HCURSOR hCursor; POINT ptScreenPos; };
struct MSG {};
inline constexpr BOOL TRUE = 1, FALSE = 0;
inline constexpr DWORD CURSOR_SHOWING = 1, CURSOR_SUPPRESSED = 2;
inline constexpr DWORD PM_NOREMOVE = 0, PM_REMOVE = 1;
inline constexpr DWORD INFINITE = 0xffffffff, WAIT_FAILED = 0xffffffff;
inline constexpr DWORD WAIT_OBJECT_0 = 0, QS_ALLINPUT = 0x4ff, MWMO_INPUTAVAILABLE = 4;
#define FAILED(value) ((value) < 0)

namespace Fake {
struct Queue { int count = 0; HCURSOR cursor = nullptr; };
struct Attachment { DWORD target; BOOL attach; bool succeeded; };
struct ShowCall { DWORD owner; BOOL show; int result; };
struct State {
	std::mutex mutex;
	std::condition_variable changed;
	int waitEntries = 0;
	bool permit = false, signaled = false, automatic = false, exited = false;
	DWORD result = WAIT_OBJECT_0;
	bool failCreate = false, failDetach = false, overlappingAttach = false;
	DWORD failAttach = 0, attached = 0, attachmentOnExit = 0;
	int messages = 0, dispatched = 0;
	std::map<DWORD, Queue> queues;
	std::vector<Attachment> attachments;
	std::vector<ShowCall> showCalls;
	std::vector<std::string> errors;
};
inline State state;

template<class Predicate>
inline void Await(std::unique_lock<std::mutex>& lock, Predicate predicate) {
	if (!state.changed.wait_for(lock, std::chrono::seconds(3), predicate)) {
		throw std::runtime_error("fake worker did not reach the expected barrier");
	}
}
inline void Reset() {
	state.waitEntries = 0;
	state.permit = state.signaled = state.automatic = state.exited = false;
	state.result = WAIT_OBJECT_0;
	state.failCreate = state.failDetach = state.overlappingAttach = false;
	state.failAttach = state.attached = state.attachmentOnExit = 0;
	state.messages = state.dispatched = 0;
	state.queues.clear();
	state.attachments.clear();
	state.showCalls.clear();
	state.errors.clear();
}
inline void AwaitStarted() {
	std::unique_lock lock(state.mutex);
	Await(lock, [] { return state.waitEntries == 1; });
}
inline void Step(DWORD result = WAIT_OBJECT_0) {
	std::unique_lock lock(state.mutex);
	const int previous = state.waitEntries;
	state.result = result;
	state.permit = true;
	state.changed.notify_all();
	Await(lock, [&] { return state.waitEntries > previous || state.exited; });
}
inline void AllowShutdown() {
	std::lock_guard lock(state.mutex);
	state.automatic = true;
	state.changed.notify_all();
}
struct ThreadLifetime {
	~ThreadLifetime() {
		std::lock_guard lock(state.mutex);
		state.attachmentOnExit = state.attached;
		state.attached = 0; // Windows removes an attachment when its thread exits.
		state.exited = true;
		state.changed.notify_all();
	}
};
}

inline DWORD GetCurrentThreadId() {
	thread_local Fake::ThreadLifetime lifetime;
	(void)lifetime;
	return 0xfeed;
}
inline DWORD GetLastError() { return 5; }
inline BOOL PeekMessage(MSG*, void*, unsigned, unsigned, DWORD remove) {
	if (remove == PM_REMOVE && Fake::state.messages > 0) {
		--Fake::state.messages;
		return TRUE;
	}
	return FALSE;
}
inline BOOL TranslateMessage(const MSG*) { return TRUE; }
inline long DispatchMessage(const MSG*) { ++Fake::state.dispatched; return 0; }
inline DWORD MsgWaitForMultipleObjectsEx(DWORD, const HANDLE*, DWORD, DWORD, DWORD) {
	auto& s = Fake::state;
	std::unique_lock lock(s.mutex);
	++s.waitEntries;
	s.changed.notify_all();
	s.changed.wait(lock, [&] { return s.permit || (s.automatic && s.signaled); });
	s.permit = s.signaled = false;
	return s.result;
}
inline BOOL AttachThreadInput(DWORD, DWORD target, BOOL attach) {
	auto& s = Fake::state;
	bool succeeded = attach ? s.failAttach != target : !s.failDetach;
	if (attach && s.attached) {
		s.overlappingAttach = true;
		succeeded = false;
	}
	s.attachments.push_back({ target, attach, succeeded });
	if (succeeded) {
		s.attached = attach ? target : 0;
	}
	return succeeded;
}
inline int ShowCursor(BOOL show) {
	auto& s = Fake::state;
	const int count = s.queues[s.attached].count += show ? 1 : -1;
	s.showCalls.push_back({ s.attached, show, count });
	return count;
}
inline HCURSOR GetCursor() { return Fake::state.queues[Fake::state.attached].cursor; }
inline BOOL GetCursorInfo(CURSORINFO* info) {
	const auto& queue = Fake::state.queues[Fake::state.attached];
	info->flags = queue.count >= 0 && queue.cursor ? CURSOR_SHOWING : 0;
	info->hCursor = info->flags ? queue.cursor : nullptr;
	return TRUE;
}

namespace fmt {
template<class... Args>
std::string format(const char* message, Args&&...) { return message; }
}

namespace wil {
enum class EventOptions { None };
class unique_event_nothrow {
public:
	int create(EventOptions) { return Fake::state.failCreate ? -1 : 0; }
	HANDLE get() const { return reinterpret_cast<HANDLE>(1); }
	void SetEvent() {
		std::lock_guard lock(Fake::state.mutex);
		Fake::state.signaled = true;
		Fake::state.changed.notify_all();
	}
};
}
