#pragma once
#include "EffectDesc.h"
#include "OnnxModelRunner.h"

namespace Magpie {

class DeviceResources;

class OnnxEffectDrawer {
public:
	// 0: not an ONNX effect, 1: valid descriptor, -1: malformed descriptor.
	static int ReadDesc(EffectDesc& desc) noexcept;
	OnnxEffectDrawer(const EffectDesc& desc, DeviceResources& resources);
	void Resize(ID3D11Texture2D* input, ID3D11Texture2D* output);
	bool Draw() noexcept;

private:
	ID3D11Device* _device;
	ID3D11DeviceContext* _context;
	std::unique_ptr<OnnxModelRunner> _runner;
	winrt::com_ptr<ID3D11Texture2D> _input, _output, _staging;
	D3D11_TEXTURE2D_DESC _inputDesc{}, _outputDesc{};
	bool _timingLogged = false;
};

}
