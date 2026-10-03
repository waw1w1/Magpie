#include "pch.h"
#include "OnnxEffectDrawer.h"
#include "CommonSharedConstants.h"
#include "DeviceResources.h"
#include "Logger.h"
#include "StrHelper.h"
#include "Win32Helper.h"
#include "LocalizationService.h"
#include "ScalingWindow.h"
#include "shaders/OnnxPreviewCS.h"
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
	winrt::check_hresult(_device->CreateComputeShader(OnnxPreviewCS, sizeof(OnnxPreviewCS), nullptr, _previewShader.put()));
	_previewSampler = resources.GetSampler(D3D11_FILTER_MIN_MAG_MIP_LINEAR, D3D11_TEXTURE_ADDRESS_CLAMP);
	if (!_previewSampler) throw std::runtime_error("Cannot create model preview sampler");
	const DWORD backendThreadId = GetCurrentThreadId();
	_worker = std::make_unique<OnnxInferenceWorker>(
		[this](std::span<const float> rgb, uint32_t width, uint32_t height, const std::function<bool()>& cancelled) {
			auto result = _runner->Run(rgb, width, height, cancelled);
			std::vector<uint8_t> rgba(result.size() / 3 * 4, 255);
			for (size_t p = 0; p < result.size() / 3; ++p) {
				if (p % 16384 == 0 && cancelled()) throw std::runtime_error("ONNX conversion cancelled");
				for (size_t c = 0; c < 3; ++c) rgba[p * 4 + c] = uint8_t(std::lround(std::clamp(result[p * 3 + c], 0.0f, 1.0f) * 255.0f));
			}
			return rgba;
		}, [this] { _runner->Cancel(); }, [backendThreadId] {
			PostThreadMessage(backendThreadId, CommonSharedConstants::WM_MODEL_READY, 0, 0);
		});
}

void OnnxEffectDrawer::Resize(ID3D11Texture2D* input, ID3D11Texture2D* output) {
	_worker->Reset();
	_hasResult = false;
	_previewAnnounced = false;
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
	_previewInput = nullptr;
	_previewOutput = nullptr;
	winrt::check_hresult(_device->CreateShaderResourceView(input, nullptr, _previewInput.put()));
	winrt::check_hresult(_device->CreateUnorderedAccessView(output, nullptr, _previewOutput.put()));
}

bool OnnxEffectDrawer::Draw() noexcept {
	try {
		const uint32_t width = _inputDesc.Width, height = _inputDesc.Height;
		// Only this rendering thread touches D3D11. The worker owns inference
		// and returns CPU pixels; results invalidated by Resize are discarded.
		auto result = _worker->TakeResult();
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
		const bool matches = result && result->width == width && result->height == height && *result->input == rgb;
		if (_worker->Submit(std::move(rgb), width, height)) _hasResult = false;
		// Never replay an old dialogue or menu over a newer game frame. While
		// content changes faster than inference, show its current live preview.
		if (matches) {
			if (result->width != width || result->height != height ||
				result->rgba.size() != size_t(_outputDesc.Width) * _outputDesc.Height * 4) {
				throw std::runtime_error("Unexpected asynchronous model result size");
			}
			_context->UpdateSubresource(_output.get(), 0, nullptr, result->rgba.data(), _outputDesc.Width * 4, 0);
			_hasResult = true;
			if (!_timingLogged) {
				Logger::Get().Info(fmt::format("ONNX first inference {}x{}: {:.2f} ms (worker time; excludes D3D11 transfers)", width, height, result->milliseconds));
				_timingLogged = true;
			}
		}
		if (!_hasResult) {
			if (!_previewAnnounced) {
				ScalingWindow::Get().ShowToast(LocalizationService::Get().GetLocalizedString(L"Message_ModelPreview"));
				_previewAnnounced = true;
			}
			ID3D11ShaderResourceView* srv = _previewInput.get();
			ID3D11UnorderedAccessView* uav = _previewOutput.get();
			_context->CSSetShader(_previewShader.get(), nullptr, 0);
			_context->CSSetShaderResources(0, 1, &srv);
			_context->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
			_context->CSSetSamplers(0, 1, &_previewSampler);
			_context->Dispatch((_outputDesc.Width + 7) / 8, (_outputDesc.Height + 7) / 8, 1);
			srv = nullptr; uav = nullptr;
			_context->CSSetShaderResources(0, 1, &srv);
			_context->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
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
