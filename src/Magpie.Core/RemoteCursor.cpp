#include "pch.h"
#include "RemoteCursor.h"
#include "Logger.h"

namespace Magpie {

RemoteCursor::RemoteCursor() noexcept {
	if (FAILED(_updateEvent.create(wil::EventOptions::None))) {
		Logger::Get().Win32Error("创建远程光标事件失败");
		return;
	}
	_thread = std::thread(&RemoteCursor::_ThreadProc, this);
}

RemoteCursor::~RemoteCursor() noexcept {
	if (!_thread.joinable()) {
		return;
	}
	{
		std::lock_guard lock(_mutex);
		_stopping = true;
	}
	_updateEvent.SetEvent();
	_thread.join();
}

void RemoteCursor::Update(DWORD threadId, CURSORINFO* cursorInfo) noexcept {
	if (!_thread.joinable()) {
		return;
	}
	{
		std::lock_guard lock(_mutex);
		if (_stopping) {
			return;
		}
		if (_requestedThreadId != threadId) {
			_requestedThreadId = threadId;
			_failedThreadId = 0;
			_hasCursorInfo = false;
		}
		if (threadId && cursorInfo && !(cursorInfo->flags & CURSOR_SUPPRESSED)) {
			if (threadId == _attachedThreadId && _hasCursorInfo) {
				// ShowCursor 隐藏后 GetCursorInfo 的 hCursor 也可能为 NULL。
				// 显隐和形状都必须从附加的输入队列读取，不能只修正 flags。
				cursorInfo->hCursor = _hCursor;
				cursorInfo->flags = _isCursorShowing && _hCursor ? CURSOR_SHOWING : 0;
			} else if (_failedThreadId != threadId) {
				// Update 只提交异步请求；新输入队列可能还未隐藏光标。
				// 等工作线程完成后再绘制，不能使用上一个队列的快照。
				cursorInfo->hCursor = NULL;
				cursorInfo->flags = 0;
			}
		}
	}
	_updateEvent.SetEvent();
}

void RemoteCursor::_ThreadProc() noexcept {
	// AttachThreadInput 要求双方都已有消息队列。
	MSG msg;
	PeekMessage(&msg, NULL, 0, 0, PM_NOREMOVE);
	const DWORD threadId = GetCurrentThreadId();
	const HANDLE updateEvent = _updateEvent.get();
	int hideCount = 0;

	auto restore = [&]() -> bool {
		_hasCursorInfo = false;
		// 只抵消本线程的调用；不把游戏的显示计数强行恢复成 0。
		while (hideCount > 0) {
			ShowCursor(TRUE);
			--hideCount;
		}
		if (_attachedThreadId) {
			if (!AttachThreadInput(threadId, _attachedThreadId, FALSE)) {
				Logger::Get().Win32Error("分离远程光标输入线程失败");
				return false;
			}
			_attachedThreadId = 0;
		}
		return true;
	};

	while (true) {
		// 附加输入队列期间也必须处理消息，不能只等待事件。
		const DWORD waitResult = MsgWaitForMultipleObjectsEx(
			1, &updateEvent, INFINITE, QS_ALLINPUT, MWMO_INPUTAVAILABLE);
		if (waitResult == WAIT_FAILED) {
			Logger::Get().Win32Error("等待远程光标事件失败");
			break;
		}
		while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}

		std::lock_guard lock(_mutex);
		if (_stopping) {
			break;
		}

		bool attached = false;
		if (_requestedThreadId != _attachedThreadId) {
			if (!restore()) {
				// 不能在旧附加关系未知时再附加其他队列。退出工作线程让系统清理。
				_stopping = true;
				return;
			}
			if (_requestedThreadId) {
				if (!AttachThreadInput(threadId, _requestedThreadId, TRUE)) {
					if (_failedThreadId != _requestedThreadId) {
						Logger::Get().Win32Error("附加远程光标输入线程失败");
					}
					_failedThreadId = _requestedThreadId;
					continue;
				}
				_attachedThreadId = _requestedThreadId;
				_failedThreadId = 0;
				attached = true;
			}
		}
		if (!_attachedThreadId) {
			continue;
		}

		CURSORINFO before{ .cbSize = sizeof(CURSORINFO) };
		const bool hasBefore = attached && GetCursorInfo(&before);
		if (attached && !hasBefore) {
			Logger::Get().Win32Error("读取远程光标隐藏前状态失败");
		}

		// 先减再加来读取共享计数，避免探测本身短暂显示系统光标。
		ShowCursor(FALSE);
		int cursorCount = ShowCursor(TRUE);
		const bool isCursorShowing = cursorCount + hideCount >= 0;
		while (cursorCount >= 0) {
			cursorCount = ShowCursor(FALSE);
			++hideCount;
		}
		// ShowCursor 不会清空输入队列中的光标形状。
		_hCursor = GetCursor();
		_isCursorShowing = isCursorShowing;
		_hasCursorInfo = true;

		if (attached) {
			CURSORINFO after{ .cbSize = sizeof(CURSORINFO) };
			if (!GetCursorInfo(&after)) {
				Logger::Get().Win32Error("读取远程光标隐藏后状态失败");
			} else if (hasBefore) {
				// 仅在附加新队列时记录，辅助排查隐藏状态与捕获结果不一致的问题。
				Logger::Get().Info(fmt::format(
					"远程光标线程 {}: count={}, hideCount={}, flags={} -> {}, hCursor={:p} -> {:p}, queueCursor={:p}",
					_attachedThreadId, cursorCount, hideCount, before.flags, after.flags,
					(void*)before.hCursor, (void*)after.hCursor, (void*)_hCursor));
			}
		}
	}

	std::lock_guard lock(_mutex);
	_stopping = true;
	restore();
}

}
