// LeRF adaptation of ddlee-cn/LeRF-PyTorch, MIT. See LICENSE.txt.
// Copyright (c) 2024 Jiacheng Li.
// Arbitrary-scale upsampling. Downscaling uses bilinear as a conservative fallback.
//!MAGPIE EFFECT
//!VERSION 4

//!TEXTURE
Texture2D INPUT;

//!TEXTURE
Texture2D OUTPUT;

//!TEXTURE
//!SOURCE LeRF-G.dds
//!FORMAT R8G8B8A8_UNORM
Texture2D LUT;

//!TEXTURE
//!WIDTH INPUT_WIDTH
//!HEIGHT INPUT_HEIGHT
//!FORMAT R32G32B32A32_FLOAT
Texture2D PREP;

//!TEXTURE
//!WIDTH INPUT_WIDTH
//!HEIGHT INPUT_HEIGHT
//!FORMAT R32G32B32A32_FLOAT
Texture2D RHO;

//!TEXTURE
//!WIDTH INPUT_WIDTH
//!HEIGHT INPUT_HEIGHT
//!FORMAT R32G32B32A32_FLOAT
Texture2D SIGMA_Y;

//!TEXTURE
//!WIDTH INPUT_WIDTH
//!HEIGHT INPUT_HEIGHT
//!FORMAT R32G32B32A32_FLOAT
Texture2D SIGMA_X;

//!SAMPLER
//!FILTER LINEAR
SamplerState SL;

//!COMMON
#include "LeRFCommon.hlsli"

//!PASS 1
//!DESC LeRF learned prefilter
//!IN INPUT, LUT
//!OUT PREP
//!BLOCK_SIZE 8
//!NUM_THREADS 64
void Pass1(uint2 blockStart, uint3 tid) {
	uint2 p = blockStart + Rmp8x8(tid.x);
	if (any(p >= GetInputSize())) return;
	float3 result = 0;
	[loop] for (uint c = 0; c < 3; ++c) {
		[loop] for (uint mode = 0; mode < 3; ++mode) {
			[loop] for (uint r = 0; r < 4; ++r) {
				result[c] += Simplex(LUT, GatherTaps(INPUT, int2(p), mode, r, c), mode).x;
			}
		}
	}
	// The prefilter sums four rotations and averages only the three modes.
	PREP[p] = float4(round(clamp(result / 3.0, 0.0, 255.0)) / 255.0, 1);
}

//!PASS 2
//!DESC LeRF learned resampling parameters
//!IN PREP, LUT
//!OUT RHO, SIGMA_Y, SIGMA_X
//!BLOCK_SIZE 8
//!NUM_THREADS 64
void Pass2(uint2 blockStart, uint3 tid) {
	uint2 p = blockStart + Rmp8x8(tid.x);
	if (any(p >= GetInputSize())) return;
	float3 red = 0, green = 0, blue = 0;
	[loop] for (uint mode = 0; mode < 3; ++mode) {
		[loop] for (uint r = 0; r < 4; ++r) {
			uint table = 3 + 2 * mode + (r % 2);
			red += Simplex(LUT, GatherTaps(PREP, int2(p), mode, r, 0), table);
			green += Simplex(LUT, GatherTaps(PREP, int2(p), mode, r, 1), table);
			blue += Simplex(LUT, GatherTaps(PREP, int2(p), mode, r, 2), table);
		}
	}
	red = round(clamp(red / 12.0 + 127.0, 0.0, 255.0)) / 255.0;
	green = round(clamp(green / 12.0 + 127.0, 0.0, 255.0)) / 255.0;
	blue = round(clamp(blue / 12.0 + 127.0, 0.0, 255.0)) / 255.0;
	RHO[p] = float4(red.x, green.x, blue.x, 1);
	SIGMA_Y[p] = float4(red.y, green.y, blue.y, 1);
	SIGMA_X[p] = float4(red.z, green.z, blue.z, 1);
}

//!PASS 3
//!DESC LeRF resampling
//!IN INPUT, PREP, RHO, SIGMA_Y, SIGMA_X
//!OUT OUTPUT
//!STYLE PS
float4 Pass3(float2 pos) {
	if (any(GetScale() < 1.0)) return INPUT.SampleLevel(SL, pos, 0);
	float2 projected = pos * GetInputSize() - 0.5;
	// Authors' evaluation uses support size 2 for both L and G.
	int2 first = int2(ceil(projected - 1.0 - 1e-6));
	float3 sum = 0, total = 0;
	[unroll] for (int y = 0; y < 2; ++y) {
		[unroll] for (int x = 0; x < 2; ++x) {
			int2 q = first + int2(x, y);
			int2 edge = clamp(q, 0, int2(GetInputSize()) - 1);
			float2 distance = projected - float2(q);
			float3 rho = RHO.Load(int3(edge, 0)).rgb * 2.0 - 1.0;
			float3 dy = SIGMA_Y.Load(int3(edge, 0)).rgb * (10.0 * distance.y);
			float3 dx = SIGMA_X.Load(int3(edge, 0)).rgb * (10.0 * distance.x);
			float3 weight = exp(-0.5 * (dy * dy - 2.0 * rho * dy * dx + dx * dx));
			// Use the same edge extension for the image and its parameters.
			sum += weight * PREP.Load(int3(edge, 0)).rgb;
			total += weight;
		}
	}
	float3 color = sum / max(total, 1e-30);
	return float4(saturate(round(color * 255.0) / 255.0), INPUT.SampleLevel(SL, pos, 0).a);
}
