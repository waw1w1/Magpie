// A clearly announced preview while the first model result is being computed.
Texture2D<float4> INPUT : register(t0);
RWTexture2D<float4> OUTPUT : register(u0);
SamplerState LINEAR_CLAMP : register(s0);

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
	uint width, height;
	OUTPUT.GetDimensions(width, height);
	if (id.x >= width || id.y >= height) return;
	OUTPUT[id.xy] = INPUT.SampleLevel(LINEAR_CLAMP, (float2(id.xy) + 0.5) / float2(width, height), 0);
}
