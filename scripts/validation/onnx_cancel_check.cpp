#include "../../src/Magpie.Core/OnnxModelRunner.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>
using namespace std::chrono_literals;

int main(int argc, char** argv) {
	if (argc != 2) return 2;
	Magpie::OnnxModelDesc desc; desc.tileSize=32; desc.overlap=8;
	Magpie::OnnxModelRunner runner(argv[1],desc,-1);
	std::vector<float> input(17*9*3,0.25f);
	std::promise<void> started,release;
	auto start=started.get_future(),gate=release.get_future();
	std::atomic<bool> rejected=false;
	std::thread thread([&] {
		bool first=true;
		try {
			runner.Run(input,17,9,[&] {
				if (first) { first=false; started.set_value(); gate.wait(); }
				return false; // Exercise ORT termination, not just our callback.
			});
		} catch (const std::exception&) { rejected=true; }
	});
	if (start.wait_for(3s)!=std::future_status::ready) std::terminate();
	runner.Cancel(); release.set_value(); thread.join();
	if (!rejected) throw std::runtime_error("ORT ignored termination");
	auto result=runner.Run(input,17,9);
	for (float v:result) if (std::abs(v-.25f)>1e-6f) throw std::runtime_error("Cannot reuse cancelled runner");
	std::cout << "PASS: ORT cancellation and subsequent runner reuse\n";
}
