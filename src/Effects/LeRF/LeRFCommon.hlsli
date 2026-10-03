// LeRF LUT evaluation adapted from ddlee-cn/LeRF-PyTorch (MIT).
// Copyright (c) 2024 Jiacheng Li. See LICENSE.txt.
// Source revision: 8099191508de171c0d7c16fb42a11c5848bb94d0

int2 RotateTap(int2 p, uint rotation) {
	// np.rot90's positive rotation, mapped back to the unrotated image.
	if (rotation == 1) return int2(-p.y, p.x);
	if (rotation == 2) return -p;
	if (rotation == 3) return int2(p.y, -p.x);
	return p;
}

float4 GatherTaps(Texture2D<float4> tex, int2 p, uint mode, uint rotation, uint channel) {
	float4 v;
	[unroll] for (uint i = 0; i < 4; ++i) {
		int2 tap = mode == 0 ? int2(i % 2, i / 2) : (mode == 1 ? int2(i, 0) : int2(i, i));
		int2 q = clamp(p + RotateTap(tap, rotation), 0, int2(GetInputSize()) - 1);
		v[i] = round(saturate(tex.Load(int3(q, 0))[channel]) * 255.0);
	}
	return v;
}

float3 ReadLut(Texture2D<float4> lut, uint4 grid, uint table) {
	uint index = ((grid.x * 17 + grid.y) * 17 + grid.z) * 17 + grid.w;
	return round(lut.Load(int3(index % 289, index / 289 + table * 289, 0)).xyz * 255.0) - 128.0;
}

float3 Simplex(Texture2D<float4> lut, float4 value, uint table) {
	uint4 grid = uint4(value) / 16;
	float4 fraction = value - float4(grid * 16);
	uint4 order = uint4(0, 1, 2, 3);
	// Four sorted fractional coordinates select five vertices of a 4D simplex.
	[unroll] for (uint i = 0; i < 3; ++i) {
		[unroll] for (uint j = i + 1; j < 4; ++j) {
			if (fraction[order[j]] > fraction[order[i]]) {
				uint t = order[i]; order[i] = order[j]; order[j] = t;
			}
		}
	}
	float3 result = (16.0 - fraction[order.x]) * ReadLut(lut, grid, table);
	[unroll] for (uint k = 0; k < 4; ++k) {
		grid[order[k]] += 1;
		float next = k < 3 ? fraction[order[min(k + 1, 3)]] : 0.0;
		result += (fraction[order[k]] - next) * ReadLut(lut, grid, table);
	}
	return result / 16.0;
}
