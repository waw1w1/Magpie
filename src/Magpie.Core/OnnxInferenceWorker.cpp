#ifdef _WIN32
#include "pch.h"
#endif
#include "OnnxInferenceWorker.h"
#include <chrono>
#include <utility>

namespace Magpie {

OnnxInferenceWorker::OnnxInferenceWorker(Inference inference, std::function<void()> cancel,
	std::function<void()> notify) : _inference(std::move(inference)), _cancel(std::move(cancel)),
	_notify(std::move(notify)), _thread(&OnnxInferenceWorker::_Run, this) {}

OnnxInferenceWorker::~OnnxInferenceWorker() { Stop(); }

bool OnnxInferenceWorker::Submit(std::vector<float> rgb, uint32_t width, uint32_t height) {
	std::lock_guard lock(_mutex);
	if (_stopping || _error) return false;
	// Completion notifications and window resizes can render the same input
	// again. Do not start an endless inference/notification cycle for static CG.
	if (_lastSubmitted && _lastSubmitted->width == width && _lastSubmitted->height == height &&
		_lastSubmitted->rgb == rgb) return false;
	_lastSubmitted = std::make_shared<Frame>(Frame{ std::move(rgb), width, height, _generation.load() });
	_pending = _lastSubmitted;
	_wake.notify_one();
	return true;
}

std::optional<OnnxInferenceWorker::Result> OnnxInferenceWorker::TakeResult() {
	std::lock_guard lock(_mutex);
	if (_error) std::rethrow_exception(_error);
	return std::exchange(_result, std::nullopt);
}

void OnnxInferenceWorker::Reset() {
	std::lock_guard lock(_mutex);
	++_generation;
	_pending.reset();
	_lastSubmitted.reset();
	_result.reset();
	_error = nullptr;
	_cancel();
}

void OnnxInferenceWorker::Stop() {
	{
		std::lock_guard lock(_mutex);
		_stopping = true;
		++_generation;
		_pending.reset();
		_cancel();
	}
	_wake.notify_one();
	if (_thread.joinable()) _thread.join();
}

void OnnxInferenceWorker::_Run() {
	while (true) {
		std::shared_ptr<const Frame> frame;
		{
			std::unique_lock lock(_mutex);
			_wake.wait(lock, [this] { return _stopping || _pending; });
			if (_stopping) return;
			frame = std::exchange(_pending, nullptr);
		}
		Result result;
		std::exception_ptr error;
		try {
			const auto start = std::chrono::steady_clock::now();
			result.rgba = _inference(frame->rgb, frame->width, frame->height, [this, generation = frame->generation] {
				return _stopping || _generation != generation;
			});
			result.width = frame->width;
			result.height = frame->height;
			result.input = std::shared_ptr<const std::vector<float>>(frame, &frame->rgb);
			result.milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
		} catch (...) {
			error = std::current_exception();
		}
		{
			std::lock_guard lock(_mutex);
			if (_stopping) return;
			if (_generation != frame->generation) continue;
			if (error) {
				_error = error;
				_pending.reset();
			} else {
				_result = std::move(result);
			}
		}
		_notify();
	}
}

}
