#include "pch.h"
#include "OnnxEffectDrawer.h"
#include "CommonSharedConstants.h"
#include "DeviceResources.h"
#include "Logger.h"
#include "StrHelper.h"
#include "Win32Helper.h"
#include <winrt/Windows.Data.Json.h>
#include <bcrypt.h>
#include <fstream>
#include <cmath>

#pragma comment(lib, "bcrypt.lib")

namespace Magpie {

int OnnxEffectDrawer::ReadDesc(EffectDesc& desc) noexcept {
	const auto path = std::filesystem::path(CommonSharedConstants::EFFECTS_DIR) /
		(StrHelper::UTF8ToUTF16(desc.name) + L".onnx.json");
	if (GetFileAttributesW(path.c_str()) == INVALID_FILE_ATTRIBUTES) return 0;
	try {
		std::string text;
		if (!Win32Helper::ReadTextFile(path.c_str(), text)) throw std::runtime_error("Cannot read model descriptor");
		const auto json = winrt::Windows::Data::Json::JsonObject::Parse(StrHelper::UTF8ToUTF16(text));
		if (json.GetNamedNumber(L"version") != 1) throw std::runtime_error("Unknown model descriptor version");
		auto integer = [&](const wchar_t* key, uint32_t min, uint32_t max) {
			const double v = json.GetNamedNumber(key);
			if (!std::isfinite(v) || v < min || v > max || std::floor(v) != v) {
				throw std::runtime_error("Invalid model descriptor integer");
			}
			return uint32_t(v);
		};
		desc.onnx.file = StrHelper::UTF16ToUTF8(json.GetNamedString(L"file"));
		desc.onnx.sha256 = StrHelper::UTF16ToUTF8(json.GetNamedString(L"sha256"));
		if (desc.onnx.file.empty() || desc.onnx.file.find_first_of("/\\:") != std::string::npos ||
			!desc.onnx.file.ends_with(".onnx") || desc.onnx.sha256.size() != 64 ||
			desc.onnx.sha256.find_first_not_of("0123456789abcdef") != std::string::npos) {
			throw std::runtime_error("Invalid model filename or SHA256");
		}
		desc.onnx.scale = integer(L"scale", 2, 4);
		desc.onnx.tileSize = integer(L"tileSize", 16, 512);
		desc.onnx.overlap = integer(L"overlap", 0, desc.onnx.tileSize / 2 - 1);
		desc.sortName = StrHelper::UTF16ToUTF8(json.GetNamedString(L"name"));
		desc.textures.resize(2);
		desc.textures[0].name = "INPUT";
		desc.textures[1].name = "OUTPUT";
		desc.textures[0].format = desc.textures[1].format = EffectIntermediateTextureFormat::R8G8B8A8_UNORM;
		desc.textures[1].sizeExpr = { fmt::format("INPUT_WIDTH * {}", desc.onnx.scale), fmt::format("INPUT_HEIGHT * {}", desc.onnx.scale) };
		EffectPassDesc pass;
		pass.desc = desc.sortName + " (DirectML; see log for wall time)";
		pass.inputs.push_back(0);
		pass.outputs.push_back(1);
		desc.passes.push_back(std::move(pass));
		return 1;
	} catch (const std::exception& e) {
		Logger::Get().Error(fmt::format("ONNX descriptor {}: {}", desc.name, e.what()));
	} catch (const winrt::hresult_error& e) {
		Logger::Get().ComError("Invalid ONNX descriptor", e.code());
	}
	return -1;
}

static void VerifyModel(const std::filesystem::path& path, const std::string& expected) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) throw std::runtime_error("Model missing. Run scripts/install_sr_models.py --effects-dir <Magpie>/effects");
	const auto size = file.tellg();
	if (size <= 0 || size > 512 * 1024 * 1024) throw std::runtime_error("Invalid model file size");
	std::vector<unsigned char> bytes(static_cast<size_t>(size));
	file.seekg(0);
	if (!file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) {
		throw std::runtime_error("Cannot read model weights");
	}
	std::array<unsigned char, 32> digest{};
	if (BCryptHash(BCRYPT_SHA256_ALG_HANDLE, nullptr, 0, bytes.data(), static_cast<ULONG>(bytes.size()),
		digest.data(), static_cast<ULONG>(digest.size())) < 0) throw std::runtime_error("Cannot hash model");
	std::string actual;
	for (auto b : digest) actual += fmt::format("{:02x}", b);
	if (actual != expected) throw std::runtime_error("Model SHA256 mismatch. Reinstall the matching weights and descriptor");
}

OnnxEffectDrawer::OnnxEffectDrawer(const EffectDesc& desc, DeviceResources& resources) :
	_device(resources.GetD3DDevice()), _context(resources.GetD3DDC()) {
	const auto directory = (std::filesystem::path(CommonSharedConstants::EFFECTS_DIR) /
		StrHelper::UTF8ToUTF16(desc.name)).parent_path();
	const auto path = directory / StrHelper::UTF8ToUTF16(desc.onnx.file);
	VerifyModel(path, desc.onnx.sha256);
	DXGI_ADAPTER_DESC selected{};
	winrt::check_hresult(resources.GetGraphicsAdapter()->GetDesc(&selected));
	int deviceIndex = -1;
	for (UINT i = 0;; ++i) {
		winrt::com_ptr<IDXGIAdapter1> adapter;
		if (resources.GetDXGIFactory()->EnumAdapters1(i, adapter.put()) == DXGI_ERROR_NOT_FOUND) break;
		if (!adapter) throw std::runtime_error("Cannot enumerate DirectML adapter");
		DXGI_ADAPTER_DESC1 candidate{};
		winrt::check_hresult(adapter->GetDesc1(&candidate));
		if (candidate.AdapterLuid.HighPart == selected.AdapterLuid.HighPart &&
			candidate.AdapterLuid.LowPart == selected.AdapterLuid.LowPart) {
			deviceIndex = static_cast<int>(i); break;
		}
	}
	if (deviceIndex < 0) throw std::runtime_error("Cannot match Magpie's adapter to DirectML");
	_runner = std::make_unique<OnnxModelRunner>(path, desc.onnx, deviceIndex);
}

void OnnxEffectDrawer::Resize(ID3D11Texture2D* input, ID3D11Texture2D* output) {
	input->GetDesc(&_inputDesc);
	output->GetDesc(&_outputDesc);
	if (_inputDesc.Format != DXGI_FORMAT_R8G8B8A8_UNORM && _inputDesc.Format != DXGI_FORMAT_B8G8R8A8_UNORM &&
		_inputDesc.Format != DXGI_FORMAT_B8G8R8X8_UNORM) throw std::runtime_error("ONNX requires SDR 8-bit RGB input");
	_input.copy_from(input); _output.copy_from(output);
	auto stagingDesc = _inputDesc;
	stagingDesc.Usage = D3D11_USAGE_STAGING;
	stagingDesc.BindFlags = 0;
	stagingDesc.MiscFlags = 0;
	stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	_staging = nullptr;
	winrt::check_hresult(_device->CreateTexture2D(&stagingDesc, nullptr, _staging.put()));
}

bool OnnxEffectDrawer::Draw() noexcept {
	try {
		const auto start = std::chrono::steady_clock::now();
		const uint32_t width = _inputDesc.Width, height = _inputDesc.Height;
		std::vector<float> rgb(size_t(width) * height * 3);
		_context->CopyResource(_staging.get(), _input.get());
		D3D11_MAPPED_SUBRESOURCE mapped{};
		winrt::check_hresult(_context->Map(_staging.get(), 0, D3D11_MAP_READ, 0, &mapped));
		const bool bgr = _inputDesc.Format != DXGI_FORMAT_R8G8B8A8_UNORM;
		for (uint32_t y = 0; y < height; ++y) {
			const auto* row = static_cast<const uint8_t*>(mapped.pData) + size_t(y) * mapped.RowPitch;
			for (uint32_t x = 0; x < width; ++x) {
				for (uint32_t c = 0; c < 3; ++c) rgb[(size_t(y) * width + x) * 3 + c] = row[x * 4 + (bgr ? 2 - c : c)] / 255.0f;
			}
		}
		_context->Unmap(_staging.get(), 0);
		auto result = _runner->Run(rgb, width, height, []() {
			MSG msg{};
			return PeekMessage(&msg, nullptr, WM_QUIT, WM_QUIT, PM_NOREMOVE) != FALSE;
		});
		std::vector<uint8_t> rgba(size_t(_outputDesc.Width) * _outputDesc.Height * 4, 255);
		for (size_t p = 0; p < rgba.size() / 4; ++p) {
			for (size_t c = 0; c < 3; ++c) rgba[p * 4 + c] = uint8_t(std::lround(std::clamp(result[p * 3 + c], 0.0f, 1.0f) * 255.0f));
		}
		_context->UpdateSubresource(_output.get(), 0, nullptr, rgba.data(), _outputDesc.Width * 4, 0);
		if (!_timingLogged) {
			const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
			Logger::Get().Info(fmt::format("ONNX first frame {}x{}: {:.2f} ms including readback, inference and upload", width, height, ms));
			_timingLogged = true;
		}
		return true;
	} catch (const std::exception& e) {
		Logger::Get().Error(fmt::format("ONNX inference failed: {}", e.what()));
	} catch (const winrt::hresult_error& e) {
		Logger::Get().ComError("ONNX texture transfer failed", e.code());
	}
	return false;
}

}
