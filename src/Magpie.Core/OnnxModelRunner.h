#pragma once
#include "include/OnnxModelDesc.h"
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace Magpie {

// Portable NCHW RGB inference and tile stitching. Windows production uses DML;
// the CPU provider exists for independent local numerical verification.
class OnnxModelRunner {
public:
	OnnxModelRunner(const std::filesystem::path& path, const OnnxModelDesc& desc, int dmlDevice);
	~OnnxModelRunner();
	OnnxModelRunner(const OnnxModelRunner&) = delete;
	OnnxModelRunner& operator=(const OnnxModelRunner&) = delete;

	std::vector<float> Run(std::span<const float> rgb, uint32_t width, uint32_t height,
		const std::function<bool()>& cancelled = {});

private:
	struct Impl;
	std::unique_ptr<Impl> _impl;
};

}
