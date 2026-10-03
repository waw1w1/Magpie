#pragma once
#include "EffectDesc.h"
#include "OnnxModelRunner.h"
#include "OnnxInferenceWorker.h"

namespace Magpie {

class DeviceResources;

class OnnxEffectDrawer {
public:
	// 0: not an ONNX effect, 1: valid descriptor, -1: malformed descriptor.
	static int ReadDesc(EffectDesc& desc) noexcept;
	OnnxEffectDrawer(const EffectDesc& desc, DeviceResources& resources);
	void Resize(ID3D11Texture2D* input, ID3D11Texture2D* output);
	bool Draw() noexcept;
	bool HasResult() const noexcept { return _hasResult; }

private:
	ID3D11Device* _device;
	ID3D11DeviceContext* _context;
	std::unique_ptr<OnnxModelRunner> _runner;
	// Destroy/join the worker before releasing its runner.
	std::unique_ptr<OnnxInferenceWorker> _worker;
	winrt::com_ptr<ID3D11Texture2D> _input, _output, _staging;
	winrt::com_ptr<ID3D11ComputeShader> _previewShader;
	winrt::com_ptr<ID3D11ShaderResourceView> _previewInput;
	winrt::com_ptr<ID3D11UnorderedAccessView> _previewOutput;
	ID3D11SamplerState* _previewSampler = nullptr;
	D3D11_TEXTURE2D_DESC _inputDesc{}, _outputDesc{};
	bool _timingLogged = false;
	bool _hasResult = false;
	bool _previewAnnounced = false;
};

}
