#pragma once
#include <mutex>
#include <thread>

namespace Magpie {

// 让通过 GetCursorInfo/DXGI 捕获光标的远程软件也能看到隐藏状态。
// 只附加无窗口的工作线程，避免把缩放窗口与源窗口的焦点、捕获状态合并。
class RemoteCursor {
public:
	RemoteCursor() noexcept;
	~RemoteCursor() noexcept;

	RemoteCursor(const RemoteCursor&) = delete;
	RemoteCursor& operator=(const RemoteCursor&) = delete;

	// threadId 为 0 时恢复。隐藏请求尚未完成时暂不绘制软件光标，避免切换时出现双光标。
	// 附加失败则保留调用方的 CURSORINFO，回落到原有光标行为。
	void Update(DWORD threadId, CURSORINFO* cursorInfo = nullptr) noexcept;

private:
	void _ThreadProc() noexcept;

	wil::unique_event_nothrow _updateEvent;
	std::thread _thread;
	std::mutex _mutex;
	DWORD _requestedThreadId = 0;
	DWORD _attachedThreadId = 0;
	DWORD _failedThreadId = 0;
	HCURSOR _hCursor = NULL;
	bool _hasCursorInfo = false;
	bool _isCursorShowing = false;
	bool _stopping = false;
};

}
