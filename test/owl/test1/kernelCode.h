#pragma once
#include <cuda.h>

#if defined(__CUDACC__)
__forceinline__ __device__ float convertRgbToL(float3 color) {
	return  0.1769f * color.x + 0.8124f * color.y + 0.0107f * color.z;
}
__forceinline__ __device__ float3 convertRgbToXyz(float3 color) {
	float x = 0.4898f * color.x + 0.3101f * color.y + 0.2001f * color.z;
	float y = 0.1769f * color.x + 0.8124f * color.y + 0.0107f * color.z;
	float z = 0.0000f * color.x + 0.0100f * color.y + 0.9903f * color.z;
	return make_float3(x, y, z);
}
__forceinline__ __device__ float3 convertXyzToRgb(float3 color) {
	float r =  2.3655f * color.x - 0.8971f * color.y - 0.4683f * color.z;
	float g = -0.5151f * color.x + 1.4264f * color.y + 0.0887f * color.z;
	float b =  0.0052f * color.x - 0.0144f * color.y + 1.0089f * color.z;
	return make_float3(r, g, b);
}
__forceinline__ __device__ float convertRgbToY(float3 color) {
	return 0.299f * color.x + 0.587f * color.y + 0.114f * color.z;
}

__forceinline__ __device__ float3 convertRgbToYCbCr(float3 color) {
	float y  =  0.299f   * color.x + 0.587f   * color.y + 0.114f   * color.z;
	float cb = -0.16874f * color.x - 0.33126f * color.y + 0.5f     * color.z;
	float cr =  0.5f     * color.x - 0.41869f * color.y - 0.08131f * color.z;
	return make_float3(y, cb, cr);
}
__forceinline__ __device__ float3 convertYCbCrToRgb(float3 color) {
	float r = color.x + 1.402f   * color.z;
	float g = color.x - 0.34414f * color.y - 0.71414f * color.z;
	float b = color.x + 1.772f   * color.y;
	return make_float3(r, g, b);
}

#endif
