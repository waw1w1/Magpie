#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <vector>

namespace Magpie {

// One running frame and one replaceable pending frame. Rendering never waits
// for inference. Reset invalidates both queued work and results from old sizes.
class OnnxInferenceWorker {
public:
	using Inference = std::function<std::vector<uint8_t>(std::span<const float>, uint32_t, uint32_t,
		const std::function<bool()>&)>;
	struct Result {
		std::vector<uint8_t> rgba;
		std::shared_ptr<const std::vector<float>> input;
		uint32_t width = 0, height = 0;
		double milliseconds = 0;
	};

	OnnxInferenceWorker(Inference inference, std::function<void()> cancel, std::function<void()> notify);
	~OnnxInferenceWorker();
	OnnxInferenceWorker(const OnnxInferenceWorker&) = delete;
	OnnxInferenceWorker& operator=(const OnnxInferenceWorker&) = delete;
	// Returns true when the displayed input changed; duplicates are not queued.
	bool Submit(std::vector<float> rgb, uint32_t width, uint32_t height);
	std::optional<Result> TakeResult();
	void Reset();
	void Stop();

private:
	struct Frame {
		std::vector<float> rgb;
		uint32_t width, height;
		uint64_t generation;
	};
	void _Run();
	Inference _inference;
	std::function<void()> _cancel, _notify;
	std::mutex _mutex;
	std::condition_variable _wake;
	std::shared_ptr<const Frame> _pending, _lastSubmitted;
	std::optional<Result> _result;
	std::exception_ptr _error;
	std::atomic<uint64_t> _generation = 0;
	std::atomic<bool> _stopping = false;
	std::thread _thread;
};

}
