#include "../../src/Magpie.Core/OnnxInferenceWorker.h"
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>
using namespace std::chrono_literals;
using Magpie::OnnxInferenceWorker;

static void Require(bool condition) {
	if (!condition) throw std::runtime_error("Worker regression check failed");
}

static void Wait(std::future<void>& event) { Require(event.wait_for(3s) == std::future_status::ready); }

int main() {
	// Hold the first job while rendering queues two replacements. Only the
	// newest pending frame may run, and a static frame must not resubmit itself.
	{
		std::promise<void> started, release, done;
		auto start = started.get_future(), gate = release.get_future(), finish = done.get_future();
		std::vector<int> seen;
		std::atomic<int> notifications = 0;
		OnnxInferenceWorker worker([&](auto rgb, auto, auto, const auto&) {
			seen.push_back(int(rgb[0]));
			if (seen.size() == 1) { started.set_value(); Wait(gate); }
			return std::vector<uint8_t>{ uint8_t(rgb[0]) };
		}, [] {}, [&] { if (++notifications == 2) done.set_value(); });
		Require(worker.Submit({1}, 1, 1)); Wait(start);
		Require(worker.Submit({2}, 1, 1));
		Require(worker.Submit({3}, 1, 1));
		Require(!worker.Submit({3}, 1, 1));
		release.set_value(); Wait(finish);
		auto result = worker.TakeResult();
		Require(result && result->rgba[0] == 3 && result->input->at(0) == 3);
		Require(!worker.TakeResult());
		worker.Stop(); Require(seen == std::vector<int>({1,3}));
	}
	// A non-cooperating old job completing after Resize must still be ignored.
	{
		std::promise<void> started, release, done;
		auto start=started.get_future(),gate=release.get_future(),finish=done.get_future();
		std::atomic<int> notifications=0;
		OnnxInferenceWorker worker([&](auto rgb, auto, auto, const auto& cancelled) {
			if (rgb[0] == 1) { started.set_value(); Wait(gate); Require(cancelled()); }
			return std::vector<uint8_t>{uint8_t(rgb[0])};
		}, [] {}, [&] { ++notifications; done.set_value(); });
		worker.Submit({1},1,1); Wait(start);
		worker.Reset(); worker.Submit({2},2,1); release.set_value(); Wait(finish);
		auto result=worker.TakeResult();
		Require(result && result->width==2 && result->rgba[0]==2 && notifications==1);
	}
	// Stop signals cancellation of an in-flight job and does not publish it.
	{
		std::promise<void> started, cancel;
		auto start=started.get_future(),gate=cancel.get_future();
		std::atomic<bool> cancelled=false;
		OnnxInferenceWorker worker([&](auto,auto,auto,const auto& stop) {
			started.set_value(); Wait(gate); Require(stop()); return std::vector<uint8_t>{};
		}, [&] { if (!cancelled.exchange(true)) cancel.set_value(); }, [] { Require(false); });
		worker.Submit({1},1,1); Wait(start); worker.Stop(); Require(!worker.TakeResult());
	}
	// Inference failure must reach the renderer, rather than leave a preview
	// running indefinitely. Reset permits a fresh request after cancellation.
	{
		std::promise<void> done; auto finish=done.get_future();
		OnnxInferenceWorker worker([](auto,auto,auto,const auto&) -> std::vector<uint8_t> {
			throw std::runtime_error("model failed");
		}, [] {}, [&] { done.set_value(); });
		worker.Submit({1},1,1); Wait(finish);
		bool failed=false;
		try { worker.TakeResult(); } catch (const std::runtime_error&) { failed=true; }
		Require(failed); worker.Reset(); Require(!worker.TakeResult());
	}
	std::cout << "PASS: nonblocking submit, latest frame, deduplication, resize invalidation, cancellation, errors\n";
}
