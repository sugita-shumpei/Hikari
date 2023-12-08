#include "tonemap_impl.h"
#include "../common.h"
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#define EPS 0.0001f

__device__ float3  getRGB2XYZ(float3 v) {
	return
	{
		0.4898f * v.x + 0.3101f * v.y + 0.2001f * v.z,
		0.1769f * v.x + 0.8124f * v.y + 0.0107f * v.z,
		0.0000f * v.x + 0.0100f * v.y + 0.9903f * v.z
	};
}

__device__ float3  getXYZ2RGB(float3 v) {
	return {
		 2.3655f * v.x - 0.8971f * v.y - 0.4683f * v.z,
		-0.5151f * v.x + 1.4264f * v.y + 0.0887f * v.z,
		 0.0052f * v.x - 0.0144f * v.y + 1.008f  * v.z
	};
}

__device__ float3  getXYZ2xyY(float3 v) {
	float denom = (v.x + v.y + v.z);
	if (denom > 0.0f) {
		float x = v.x / denom;
		float y = v.y / denom;
		return { x,y,v.y };
	}
	else {
		return { 0.0f,0.0f,0.0f };
	}
}

__device__ float3  getxyY2XYZ(float3 v) {
	float Y = v.z;
	if (Y > 0.0f) {
		float X = (v.x / v.y) * Y;
		float Z = (Y / v.y) * (1.0f - v.x - v.y);
		return { X ,Y,Z};
	}
	else {
		return { 0.0f,0.0f,0.0f };
	}
}

__device__ float  getLuminance(float3 v) {
	return 0.1769f * v.x + 0.8124f* v.y + 0.0107f * v.z;
}
__device__ float3 changeLuminance(float3 v,float luminance_out) {
	float luminance_in = getLuminance(v);
	float l_c          = luminance_in > 0.0f ? luminance_out / luminance_in : 0.0f;
	float v_max        = fmaxf(v.x,fmaxf(v.y,v.z));
	float l_th         = (v_max > 0.0f) ? 1.0f/v_max:1.0f;
	l_c                = fminf(l_th,l_c);
	return { v.x * l_c,v.y * l_c ,v.z * l_c   };
}
__device__ float3 reinhard(float3 v, float luminance, float max_luminance) {
	float l_old = luminance;
	float l_new = l_old  / (l_old + 1.0f);
	return changeLuminance(v, l_new);
}
__device__ float3 extendedReinhard(float3 v, float luminance, float max_luminance) {
	float l_old = luminance;
	float l_new = l_old * (1.0f + (l_old / (max_luminance * max_luminance))) / (l_old + 1.0f);
	return changeLuminance(v, l_new);
}

__global__ void HikariTestOwlTonemap_estimateLuminanceImpl(int width, int height, const float3* input_buffer, float* luminance_buffer, float* luminance_log_buffer)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) { return; }
	auto input_color = input_buffer[width * y + x];
	auto luminance   = getLuminance(input_color);
	luminance_buffer[width * y + x]     = luminance;
	luminance_log_buffer[width * y + x] = logf(EPS + luminance);
}

__global__ void HikariTestOwlTonemap_tonemapColorRGBA8Impl_linear(
	int           width,
	int           height,
	const float3* input_buffer,
	unsigned int* output_buffer,
	float         max_luminance,
	float         ave_luminance,
	float         key_value)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) { return; }
	auto input_color = input_buffer[width * y + x];
	unsigned int r   = fminf(255, fmaxf(0, int(input_color.x * 256.f)));
	unsigned int g   = fminf(255, fmaxf(0, int(input_color.y * 256.f)));
	unsigned int b   = fminf(255, fmaxf(0, int(input_color.z * 256.f)));
	output_buffer[width * y + x] = (r << 0u) + (g << 8u) + (b << 16u) + (0xffu << 24u);
}

__global__ void HikariTestOwlTonemap_tonemapColorRGBA8Impl_correlatedLinear(
	int           width,
	int           height,
	const float3* input_buffer,
	unsigned int* output_buffer,
	float         max_luminance,
	float         ave_luminance,
	float         key_value)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) { return; }
	auto input_color   = input_buffer[width * y + x];
	auto luminance     = getLuminance(input_color);
	auto crr_luminance = key_value * luminance / ave_luminance;
	input_color        = changeLuminance(input_color, crr_luminance);
	unsigned int r = fminf(255, fmaxf(0, int(input_color.x * 256.f)));
	unsigned int g = fminf(255, fmaxf(0, int(input_color.y * 256.f)));
	unsigned int b = fminf(255, fmaxf(0, int(input_color.z * 256.f)));
	output_buffer[width * y + x] = (r << 0u) + (g << 8u) + (b << 16u) + (0xffu << 24u);
}

__global__ void HikariTestOwlTonemap_tonemapColorRGBA8Impl_correlatedReinhard(
	int           width , 
	int           height, 
	const float3* input_buffer, 
	unsigned int* output_buffer, 
	float         max_luminance, 
	float         ave_luminance,
	float         key_value)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) { return; }
	 auto input_color   = input_buffer[width * y + x];
	 auto luminance         = getLuminance(input_color);
	 auto crr_luminance     = key_value *     luminance / ave_luminance;
	 auto crr_max_luminance = key_value * max_luminance / ave_luminance;
	 input_color = reinhard(input_color, crr_luminance, crr_max_luminance);
	unsigned int r = fminf(255, fmaxf(0, int(input_color.x * 256.f)));
	unsigned int g = fminf(255, fmaxf(0, int(input_color.y * 256.f)));
	unsigned int b = fminf(255, fmaxf(0, int(input_color.z * 256.f)));
	output_buffer[width * y + x] = (r << 0u) + (g << 8u) + (b << 16u) + (0xffu << 24u);
}

__global__ void HikariTestOwlTonemap_tonemapColorRGBA8Impl_correlatedExtendedReinhard(
	int           width,
	int           height,
	const float3* input_buffer,
	unsigned int* output_buffer,
	float         max_luminance,
	float         ave_luminance,
	float         key_value)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) { return; }
	auto input_color   = input_buffer[width * y + x];
	auto luminance         = getLuminance(input_color);
	auto crr_luminance     = key_value * luminance / ave_luminance;
	auto crr_max_luminance = key_value * max_luminance / ave_luminance;
	input_color            = extendedReinhard(input_color, crr_luminance, crr_max_luminance);
	unsigned int r         = fminf(255, fmaxf(0, int(input_color.x * 256.f)));
	unsigned int g         = fminf(255, fmaxf(0, int(input_color.y * 256.f)));
	unsigned int b         = fminf(255, fmaxf(0, int(input_color.z * 256.f)));
	output_buffer[width * y + x] = (r << 0u) + (g << 8u) + (b << 16u) + (0xffu << 24u);
}

void HikariTestOwlTonemap_estimateLuminance(cudaStream_t  stream, int width, int height, const float3* input_buffer, float* luminance_buffer, float* luminance_log_buffer)
{
	const int block_size_x = 32;
	const int block_size_y = 32;
	const int grid_size_x  = (width  + block_size_x - 1) / block_size_x;
	const int grid_size_y  = (height + block_size_y - 1) / block_size_y;

	dim3 grid(grid_size_x, grid_size_y, 1);
	dim3 threads(block_size_x, block_size_y, 1);
	HikariTestOwlTonemap_estimateLuminanceImpl << <grid, threads , 0, stream>> > (width, height, input_buffer, luminance_buffer, luminance_log_buffer);
}

void HikariTestOwlTonemap_estimateMaxAndAverage(cudaStream_t  stream, int width, int height, const float* luminance_buffer, const float* luminance_log_buffer, float* p_max_luminance, float* p_ave_luminance)
{
	auto luminance_buffer_ptr     = thrust::device_pointer_cast(luminance_buffer);
	auto luminance_log_buffer_ptr = thrust::device_pointer_cast(luminance_log_buffer);
	auto  max_iter                = thrust::max_element(thrust::cuda::par.on(stream),luminance_buffer_ptr, luminance_buffer_ptr + width * height);
	float max_value               = 0.0f;
	cudaMemcpyAsync(&max_value, max_iter.get(), sizeof(float), cudaMemcpyDeviceToHost,stream);
	auto ave_value = expf(thrust::reduce(thrust::cuda::par.on(stream), luminance_log_buffer_ptr, luminance_log_buffer_ptr + width * height) / (width * height));
	if (p_ave_luminance) *p_ave_luminance = ave_value;
	if (p_max_luminance) *p_max_luminance = max_value;
}

void HikariTestOwlTonemap_tonemapColorRGBA8(cudaStream_t  stream, int width, int height,int type, const float3* input_buffer, unsigned int* output_buffer, float max_luminance, float ave_luminance, float key_value)
{
	const int block_size_x = 32;
	const int block_size_y = 32;
	const int grid_size_x  = (width  + block_size_x - 1) / block_size_x;
	const int grid_size_y  = (height + block_size_y - 1) / block_size_y;
	
	dim3 grid(grid_size_x, grid_size_y,1);
	dim3 threads(block_size_x, block_size_y, 1);
	if (type == static_cast<int>(hikari::test::owl::testlib::TonemapType::eLinear)) {
		HikariTestOwlTonemap_tonemapColorRGBA8Impl_linear << <grid, threads, 0, stream >> > (width, height, input_buffer, output_buffer, max_luminance, ave_luminance, key_value);
	}
	if (type == static_cast<int>(hikari::test::owl::testlib::TonemapType::eCorrelatedLinear)) {
		HikariTestOwlTonemap_tonemapColorRGBA8Impl_correlatedLinear<< <grid, threads, 0, stream >> > (width, height, input_buffer, output_buffer, max_luminance, ave_luminance, key_value);
	}
	if (type == static_cast<int>(hikari::test::owl::testlib::TonemapType::eCorrelatedReinhard)) {
		HikariTestOwlTonemap_tonemapColorRGBA8Impl_correlatedReinhard<< <grid, threads, 0, stream >> > (width, height, input_buffer, output_buffer, max_luminance, ave_luminance, key_value);
	}
	if (type == static_cast<int>(hikari::test::owl::testlib::TonemapType::eCorrelatedExtendedReinhard)) {// key_value*max_luminance/ave_luminance=1
		HikariTestOwlTonemap_tonemapColorRGBA8Impl_correlatedExtendedReinhard << <grid, threads, 0, stream >> > (width, height, input_buffer, output_buffer, max_luminance, ave_luminance, key_value);
	}
}
