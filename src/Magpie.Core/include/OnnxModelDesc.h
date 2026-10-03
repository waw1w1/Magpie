#pragma once
#include <cstdint>
#include <string>

namespace Magpie {

struct OnnxModelDesc {
	std::string file;
	std::string sha256;
	uint32_t scale = 2;
	// Network input extent, including context on both sides.
	uint32_t tileSize = 128;
	uint32_t overlap = 32;
};

}
