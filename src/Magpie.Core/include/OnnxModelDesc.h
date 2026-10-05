#pragma once
#include <cstdint>
#include <string>

namespace Magpie {

enum class OnnxBackend { DirectML, TensorRtRtx };

struct OnnxModelDesc {
	std::string file;
	std::string sha256;
	OnnxBackend backend = OnnxBackend::DirectML;
	uint32_t scale = 2;
	// Network input extent, including context on both sides.
	uint32_t tileSize = 128;
	uint32_t overlap = 32;
};

}
