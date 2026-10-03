#pragma once
#include <array>
#include <string_view>

namespace Magpie {

// Stable effect IDs, independent of directory order or display names. Effects
// outside this list remain available in the non-recommended group.
inline constexpr std::array RecommendedEffects{
	std::wstring_view(L"ACNet2\\ACNet-F8B4"),
	std::wstring_view(L"ACNet2\\ACNet-F8B8"),
	std::wstring_view(L"ARNet\\ARNet-F8B8"),
	std::wstring_view(L"ArtCNN\\ArtCNN-C4F16"),
	std::wstring_view(L"ArtCNN\\ArtCNN-C4F32"),
	std::wstring_view(L"ONNX\\AnimeSharpV4"),
	std::wstring_view(L"ONNX\\AnimeSharpV4-Fast"),
	std::wstring_view(L"ONNX\\IllustrationJaNai-DAT2"),
	std::wstring_view(L"LeRF\\LeRF-L"),
	std::wstring_view(L"LeRF\\LeRF-G"),
	std::wstring_view(L"CuNNy2\\CuNNy-veryfast-NVL"),
	std::wstring_view(L"CuNNy2\\CuNNy-fast-NVL"),
	std::wstring_view(L"CuNNy2\\CuNNy-4x12-NVL"),
	std::wstring_view(L"CuNNy2\\CuNNy-4x16-NVL"),
	std::wstring_view(L"Ani4Kv2_ArtCNN_C4F32_i2"),
	std::wstring_view(L"k7_modernAnime_FHD_x2"),
	std::wstring_view(L"RAVU\\RAVU_Lite_AR_R2"),
	std::wstring_view(L"RAVU\\RAVU_Zoom_AR_R3"),
	std::wstring_view(L"FSR\\FSR_EASU"),
	std::wstring_view(L"FSR\\FSR_RCAS"),
	std::wstring_view(L"NIS\\NIS"),
	std::wstring_view(L"SGSR"),
	std::wstring_view(L"Nearest"),
	std::wstring_view(L"Bilinear"),
	std::wstring_view(L"Bicubic"),
	std::wstring_view(L"Lanczos"),
	std::wstring_view(L"Pixel Art\\SharpBilinear"),
	std::wstring_view(L"Pixel Art\\Pixellate"),
	std::wstring_view(L"SSimDownscaler")
};

constexpr bool IsRecommendedEffect(std::wstring_view name) noexcept {
	for (std::wstring_view candidate : RecommendedEffects) {
		if (name == candidate) {
			return true;
		}
	}
	return false;
}

}
