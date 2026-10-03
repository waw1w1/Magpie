#ifdef _WIN32
#include "pch.h"
#endif
#include "OnnxModelRunner.h"
#include <onnxruntime_cxx_api.h>
#ifdef _WIN32
#include <dml_provider_factory.h>
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace Magpie {

struct OnnxModelRunner::Impl {
	OnnxModelDesc desc;
	Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "Magpie" };
	Ort::Session session{ nullptr };
	std::string inputName, outputName;
	ONNXTensorElementDataType inputType{}, outputType{};

	Impl(const std::filesystem::path& path, const OnnxModelDesc& config, int dmlDevice) : desc(config) {
		if (desc.scale < 2 || desc.scale > 4 || desc.tileSize < 16 || desc.tileSize > 512 ||
			desc.overlap >= desc.tileSize / 2) {
			throw std::invalid_argument("Invalid ONNX scale or tile configuration");
		}
		Ort::SessionOptions options;
		options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
		options.DisableMemPattern();
#ifdef _WIN32
		if (dmlDevice < 0) throw std::invalid_argument("A DirectML adapter is required");
		const OrtDmlApi* dml = nullptr;
		Ort::ThrowOnError(Ort::GetApi().GetExecutionProviderApi("DML", ORT_API_VERSION,
			reinterpret_cast<const void**>(&dml)));
		Ort::ThrowOnError(dml->SessionOptionsAppendExecutionProvider_DML(options, dmlDevice));
		// ORT may place shape/control operators on CPU. A failed DML provider
		// initialization is an error; it never silently selects another adapter.
#else
		if (dmlDevice >= 0) throw std::invalid_argument("DirectML requires Windows");
		options.SetIntraOpNumThreads(2);
#endif
		session = Ort::Session(env, path.c_str(), options);
		if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) {
			throw std::runtime_error("Model must have exactly one RGB input and output");
		}
		Ort::AllocatorWithDefaultOptions allocator;
		inputName = session.GetInputNameAllocated(0, allocator).get();
		outputName = session.GetOutputNameAllocated(0, allocator).get();
		// Keep the owning TypeInfo alive while inspecting its unowned tensor info.
		auto inType = session.GetInputTypeInfo(0);
		auto outType = session.GetOutputTypeInfo(0);
		auto inInfo = inType.GetTensorTypeAndShapeInfo();
		auto outInfo = outType.GetTensorTypeAndShapeInfo();
		inputType = inInfo.GetElementType();
		outputType = outInfo.GetElementType();
		auto supported = [](auto type) {
			return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
		};
		if (!supported(inputType) || !supported(outputType)) {
			throw std::runtime_error("Only FP32/FP16 models are supported");
		}
		auto validate = [](const std::vector<int64_t>& shape, int64_t size) {
			return shape.size() == 4 && (shape[0] == 1 || shape[0] == -1) && shape[1] == 3 &&
				(shape[2] == size || shape[2] == -1) && (shape[3] == size || shape[3] == -1);
		};
		if (!validate(inInfo.GetShape(), desc.tileSize) ||
			!validate(outInfo.GetShape(), int64_t(desc.tileSize) * desc.scale)) {
			throw std::runtime_error("Model dimensions do not match the NCHW RGB preset");
		}
	}
};

OnnxModelRunner::OnnxModelRunner(const std::filesystem::path& path, const OnnxModelDesc& desc, int dmlDevice) :
	_impl(std::make_unique<Impl>(path, desc, dmlDevice)) {}

OnnxModelRunner::~OnnxModelRunner() = default;

std::vector<float> OnnxModelRunner::Run(std::span<const float> rgb, uint32_t width, uint32_t height,
	const std::function<bool()>& cancelled) {
	const auto& config = _impl->desc;
	const uint32_t scale = config.scale, tile = config.tileSize, border = config.overlap;
	if (width == 0 || height == 0 || width > 16384 / scale || height > 16384 / scale ||
		rgb.size() != size_t(width) * height * 3) {
		throw std::invalid_argument("Invalid input size for ONNX scaling");
	}
	const uint32_t stride = tile - 2 * border;
	const uint32_t outWidth = width * scale, outTile = tile * scale;
	const size_t area = size_t(tile) * tile;
	const size_t outArea = size_t(outTile) * outTile;
	std::vector<float> result(size_t(outWidth) * height * scale * 3);
	std::vector<float> input(area * 3);
	std::vector<Ort::Float16_t> halfInput;
	if (_impl->inputType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) halfInput.resize(input.size());
	const std::array<int64_t, 4> shape{ 1, 3, tile, tile };
	const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	const char* inName = _impl->inputName.c_str();
	const char* outName = _impl->outputName.c_str();

	for (uint32_t top = 0; top < height; top += stride) {
		for (uint32_t left = 0; left < width; left += stride) {
			if (cancelled && cancelled()) throw std::runtime_error("ONNX inference cancelled");
			for (uint32_t y = 0; y < tile; ++y) {
				const uint32_t sy = uint32_t(std::clamp(int64_t(top) + y - border, int64_t(0), int64_t(height - 1)));
				for (uint32_t x = 0; x < tile; ++x) {
					const uint32_t sx = uint32_t(std::clamp(int64_t(left) + x - border, int64_t(0), int64_t(width - 1)));
					for (size_t c = 0; c < 3; ++c) {
						input[c * area + size_t(y) * tile + x] = rgb[(size_t(sy) * width + sx) * 3 + c];
					}
				}
			}
			Ort::Value tensor{ nullptr };
			if (halfInput.empty()) {
				tensor = Ort::Value::CreateTensor<float>(memory, input.data(), input.size(), shape.data(), shape.size());
			} else {
				std::transform(input.begin(), input.end(), halfInput.begin(), [](float v) { return Ort::Float16_t(v); });
				tensor = Ort::Value::CreateTensor<Ort::Float16_t>(memory, halfInput.data(), halfInput.size(), shape.data(), shape.size());
			}
			auto outputs = _impl->session.Run(Ort::RunOptions{ nullptr }, &inName, &tensor, 1, &outName, 1);
			const auto info = outputs[0].GetTensorTypeAndShapeInfo();
			const std::vector<int64_t> expected{ 1, 3, outTile, outTile };
			if (info.GetShape() != expected || info.GetElementType() != _impl->outputType) {
				throw std::runtime_error("Unexpected ONNX output shape or type");
			}
			const float* fp32 = _impl->outputType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
				? outputs[0].GetTensorData<float>() : nullptr;
			const Ort::Float16_t* fp16 = fp32 ? nullptr : outputs[0].GetTensorData<Ort::Float16_t>();
			const uint32_t rows = std::min(stride, height - top) * scale;
			const uint32_t cols = std::min(stride, width - left) * scale;
			for (uint32_t y = 0; y < rows; ++y) {
				for (uint32_t x = 0; x < cols; ++x) {
					const size_t src = size_t(y + border * scale) * outTile + x + border * scale;
					const size_t dst = (size_t(top * scale + y) * outWidth + left * scale + x) * 3;
					for (size_t c = 0; c < 3; ++c) {
						const float value = fp32 ? fp32[c * outArea + src] : fp16[c * outArea + src].ToFloat();
						if (!std::isfinite(value)) throw std::runtime_error("Non-finite ONNX output");
						result[dst + c] = value;
					}
				}
			}
		}
	}
	return result;
}

}
