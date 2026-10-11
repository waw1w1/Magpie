#include "pch.h"
#include "RemoteCursor.h"
#include <iostream>
#include <memory>

namespace {
constexpr DWORD OWNER_A = 101, OWNER_B = 202;
HCURSOR Cursor(std::uintptr_t value) { return reinterpret_cast<HCURSOR>(value); }

#define CHECK(condition) do { \
	if (!(condition)) throw std::runtime_error(std::string(#condition) + " at line " + std::to_string(__LINE__)); \
} while (false)

CURSORINFO Input(DWORD flags = CURSOR_SHOWING, HCURSOR cursor = Cursor(0xbeef)) {
	return { sizeof(CURSORINFO), flags, cursor, { 17, 29 } };
}
void Unchanged(const CURSORINFO& actual, const CURSORINFO& expected) {
	CHECK(actual.cbSize == expected.cbSize);
	CHECK(actual.flags == expected.flags);
	CHECK(actual.hCursor == expected.hCursor);
	CHECK(actual.ptScreenPos.x == expected.ptScreenPos.x);
	CHECK(actual.ptScreenPos.y == expected.ptScreenPos.y);
}
void Pending(const CURSORINFO& info) {
	CHECK(info.flags == 0);
	CHECK(info.hCursor == nullptr);
	CHECK(info.cbSize == sizeof(CURSORINFO));
	CHECK(info.ptScreenPos.x == 17 && info.ptScreenPos.y == 29);
}

struct Fixture {
	std::unique_ptr<Magpie::RemoteCursor> remote;
	explicit Fixture(bool failCreate = false) {
		Fake::Reset();
		Fake::state.failCreate = failCreate;
		Fake::state.queues[OWNER_A] = { 0, Cursor(0xa) };
		Fake::state.queues[OWNER_B] = { 0, Cursor(0xb) };
		remote = std::make_unique<Magpie::RemoteCursor>();
		if (!failCreate) Fake::AwaitStarted();
	}
	~Fixture() { Stop(); }
	void Stop() {
		Fake::AllowShutdown();
		remote.reset();
	}
	void Ready(DWORD owner = OWNER_A) {
		remote->Update(owner);
		Fake::Step();
		CHECK(Fake::state.attached == owner);
	}
};

void EventCreationFailure() {
	Fixture f(true);
	auto info = Input();
	const auto original = info;
	f.remote->Update(OWNER_A, &info);
	Unchanged(info, original);
	CHECK(!Fake::state.errors.empty());
	CHECK(Fake::state.attachments.empty());
}

void PendingReadySuppressedAndMessages() {
	Fixture f;
	auto info = Input();
	f.remote->Update(OWNER_A, &info);
	Pending(info);
	CHECK(Fake::state.attachments.empty());
	const auto suppressed = Input(CURSOR_SUPPRESSED, Cursor(0xc));
	info = suppressed;
	f.remote->Update(OWNER_A, &info);
	Unchanged(info, suppressed); // Suppression also survives an unprocessed request.
	Fake::Step();
	CHECK(Fake::state.queues[OWNER_A].count == -1);
	info = Input(0, nullptr); // GetCursorInfo may lose the shape when hidden.
	f.remote->Update(OWNER_A, &info);
	CHECK(info.flags == CURSOR_SHOWING && info.hCursor == Cursor(0xa));
	info = suppressed;
	f.remote->Update(OWNER_A, &info);
	Unchanged(info, suppressed);
	Fake::state.messages = 2;
	Fake::state.queues[OWNER_A].cursor = Cursor(0xab);
	Fake::Step(WAIT_OBJECT_0 + 1); // Input-queue wake, rather than event wake.
	CHECK(Fake::state.dispatched == 2);
	CHECK(Fake::state.queues[OWNER_A].count == -1);
	info = Input(0, nullptr);
	f.remote->Update(OWNER_A, &info);
	CHECK(info.flags == CURSOR_SHOWING && info.hCursor == Cursor(0xab));
	Fake::state.queues[OWNER_A].cursor = nullptr;
	Fake::Step();
	info = Input();
	f.remote->Update(OWNER_A, &info);
	CHECK(info.flags == 0 && info.hCursor == nullptr);
	f.Stop();
	CHECK(Fake::state.queues[OWNER_A].count == 0);
	CHECK(Fake::state.attached == 0);
}

void OwnerSwitch() {
	Fixture f;
	f.Ready();
	auto info = Input();
	f.remote->Update(OWNER_B, &info);
	Pending(info);
	CHECK(Fake::state.attached == OWNER_A);
	Fake::Step();
	CHECK(Fake::state.queues[OWNER_A].count == 0);
	CHECK(Fake::state.queues[OWNER_B].count == -1);
	CHECK(!Fake::state.overlappingAttach);
	CHECK(Fake::state.attachments.size() == 3);
	CHECK(!Fake::state.attachments[1].attach && Fake::state.attachments[1].target == OWNER_A);
	CHECK(Fake::state.attachments[2].attach && Fake::state.attachments[2].target == OWNER_B);
	info = Input(0, nullptr);
	f.remote->Update(OWNER_B, &info);
	CHECK(info.flags == CURSOR_SHOWING && info.hCursor == Cursor(0xb));
}

void CancelBeforeAttachAndRestore() {
	Fixture f;
	f.remote->Update(OWNER_A);
	auto restoring = Input();
	const auto original = restoring;
	f.remote->Update(0, &restoring);
	Unchanged(restoring, original);
	Fake::Step();
	CHECK(Fake::state.attachments.empty());
	CHECK(Fake::state.showCalls.empty());
	f.Ready();
	f.remote->Update(0);
	Fake::Step();
	CHECK(Fake::state.attached == 0);
	CHECK(Fake::state.queues[OWNER_A].count == 0);
	auto info = Input();
	f.remote->Update(OWNER_A, &info);
	Pending(info);
}

void SwitchingBackCannotReuseSnapshot() {
	Fixture f;
	f.Ready();
	f.remote->Update(OWNER_B);
	auto info = Input();
	f.remote->Update(OWNER_A, &info);
	Pending(info);
	Fake::Step();
	info = Input(0, nullptr);
	f.remote->Update(OWNER_A, &info);
	CHECK(info.flags == CURSOR_SHOWING && info.hCursor == Cursor(0xa));
	f.remote->Update(0);
	info = Input();
	f.remote->Update(OWNER_A, &info);
	Pending(info); // Cancellation also invalidates a previously good snapshot.
}

void AttachFailurePreservesInput() {
	Fixture f;
	Fake::state.failAttach = OWNER_A;
	f.remote->Update(OWNER_A);
	Fake::Step();
	CHECK(Fake::state.attached == 0);
	CHECK(Fake::state.showCalls.empty());
	CHECK(!Fake::state.errors.empty());
	auto info = Input();
	const auto original = info;
	f.remote->Update(OWNER_A, &info);
	Unchanged(info, original);
	const auto errors = Fake::state.errors.size();
	Fake::Step(); // Retrying a failed owner does not repeatedly log its failure.
	CHECK(Fake::state.errors.size() == errors);
	Fake::state.failAttach = 0;
	f.remote->Update(OWNER_A);
	Fake::Step();
	info = Input(0, nullptr);
	f.remote->Update(OWNER_A, &info);
	CHECK(info.flags == CURSOR_SHOWING && info.hCursor == Cursor(0xa));
}

void DetachFailureCannotAttachAnotherOwner() {
	Fixture f;
	f.Ready();
	Fake::state.failDetach = true;
	f.remote->Update(OWNER_B);
	Fake::Step();
	CHECK(Fake::state.exited);
	CHECK(Fake::state.attachmentOnExit == OWNER_A);
	CHECK(Fake::state.queues[OWNER_A].count == 0);
	CHECK(Fake::state.queues[OWNER_B].count == 0);
	CHECK(!Fake::state.overlappingAttach);
	CHECK(!Fake::state.errors.empty());
	auto info = Input();
	const auto original = info;
	f.remote->Update(OWNER_B, &info);
	Unchanged(info, original);
	f.remote->Update(OWNER_A, &info); // The failed detach retains the old owner ID.
	Unchanged(info, original);
	CHECK(Fake::state.queues[OWNER_A].count == 0);
	CHECK(!Fake::state.overlappingAttach);
	for (const auto& call : Fake::state.attachments) {
		CHECK(!(call.attach && call.target == OWNER_B));
	}
	Fake::state.failDetach = false;
}

void WaitFailureInvalidatesSnapshot() {
	Fixture f;
	f.Ready();
	Fake::Step(WAIT_FAILED);
	CHECK(Fake::state.exited);
	CHECK(Fake::state.attached == 0);
	CHECK(Fake::state.queues[OWNER_A].count == 0);
	CHECK(!Fake::state.errors.empty());
	auto info = Input();
	const auto original = info;
	f.remote->Update(OWNER_A, &info);
	Unchanged(info, original);
}

void RestoreOnlyOwnedHideCalls() {
	for (int originalCount : { -2, 0, 3 }) {
		Fixture f;
		Fake::state.queues[OWNER_A].count = originalCount;
		f.Ready();
		CHECK(Fake::state.queues[OWNER_A].count == (originalCount < 0 ? originalCount : -1));
		auto info = Input(0, nullptr);
		f.remote->Update(OWNER_A, &info);
		CHECK(info.flags == (originalCount >= 0 ? CURSOR_SHOWING : 0));
		CHECK(info.hCursor == Cursor(0xa));
		// The source calls ShowCursor(FALSE) twice while the worker owns its hide.
		Fake::state.queues[OWNER_A].count -= 2;
		Fake::Step();
		info = Input();
		f.remote->Update(OWNER_A, &info);
		CHECK(info.flags == (originalCount - 2 >= 0 ? CURSOR_SHOWING : 0));
		// The source then calls ShowCursor(TRUE) five times; refresh may need more hides.
		Fake::state.queues[OWNER_A].count += 5;
		Fake::Step();
		info = Input(0, nullptr);
		f.remote->Update(OWNER_A, &info);
		CHECK(info.flags == CURSOR_SHOWING);
		f.Stop();
		CHECK(Fake::state.queues[OWNER_A].count == originalCount + 3);
		int ownedDelta = 0;
		for (const auto& call : Fake::state.showCalls) {
			CHECK(call.owner == OWNER_A);
			ownedDelta += call.show ? 1 : -1;
		}
		CHECK(ownedDelta == 0);
	}
}
}

int main() {
	const std::pair<const char*, void(*)()> cases[] = {
		{ "event creation failure preserves input", EventCreationFailure },
		{ "pending / ready / suppressed / message pump", PendingReadySuppressedAndMessages },
		{ "owner switch restores then attaches", OwnerSwitch },
		{ "cancel before attachment and restore", CancelBeforeAttachAndRestore },
		{ "switching back cannot reuse stale snapshot", SwitchingBackCannotReuseSnapshot },
		{ "attach failure preserves input", AttachFailurePreservesInput },
		{ "detach failure cannot attach another owner", DetachFailureCannotAttachAnotherOwner },
		{ "wait failure invalidates snapshot", WaitFailureInvalidatesSnapshot },
		{ "restore only owned hides, retaining source deltas", RestoreOnlyOwnedHideCalls },
	};
	int failed = 0;
	for (auto [name, test] : cases) {
		try {
			test();
			std::cout << "PASS " << name << '\n';
		} catch (const std::exception& error) {
			++failed;
			std::cerr << "FAIL " << name << ": " << error.what() << '\n';
		}
	}
	std::cout << std::size(cases) - failed << '/' << std::size(cases) << " passed\n";
	return failed ? 1 : 0;
}
