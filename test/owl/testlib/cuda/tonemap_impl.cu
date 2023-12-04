#include "tonemap_impl.h"
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#define EPS 0.0001f

__global__ void HikariTestOwlTonemap_estimateLuminanceImpl(int width, int height, const float3* input_buffer, float* luminance_buffer, float* luminance_log_buffer)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) { return; }
	auto input_color = input_buffer[width * y + x];
	auto luminance   = 0.2126f * input_color.x + 0.7152f * input_color.y + 0.0722f * input_color.z;
	luminance_buffer[width * y + x]     = luminance;
	luminance_log_buffer[width * y + x] = logf(EPS + luminance);
}

__global__ void HikariTestOwlTonemap_tonemapColorRGBA8Impl(
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
	auto input_color       = input_buffer[width * y + x];
	// auto luminance         = 0.2126f * input_color.x + 0.7152f * input_color.y + 0.0722f * input_color.z;
	// auto crr_luminance     = key_value *     luminance / ave_luminance;
	// auto crr_max_luminance = key_value * max_luminance / ave_luminance;
	input_color.x *= (key_value / ave_luminance);
	input_color.y *= (key_value / ave_luminance);
	input_color.z *= (key_value / ave_luminance);
	input_color.x = input_color.x / (1.0f + input_color.x);
	input_color.y = input_color.y / (1.0f + input_color.y);
	input_color.z = input_color.z / (1.0f + input_color.z);
	unsigned int r = fminf(255, fmaxf(0, int(input_color.x * 256.f)));
	unsigned int g = fminf(255, fmaxf(0, int(input_color.y * 256.f)));
	unsigned int b = fminf(255, fmaxf(0, int(input_color.z * 256.f)));
	output_buffer[width * y + x] = (r << 0u) + (g << 8u) + (b << 16u) + (0xffu << 24u);
}

void HikariTestOwlTonemap_estimateLuminance(cudaStream_t  stream, int width, int height, const float3* input_buffer, float* luminance_buffer, float* luminance_log_buffer)
{
	const int block_size_x = 32;
	const int block_size_y = 32;
	const int grid_size_x = (width + block_size_x - 1) / block_size_x;
	const int grid_size_y = (height + block_size_y - 1) / block_size_y;

	dim3 grid(grid_size_x, grid_size_y, 1);
	dim3 threads(block_size_x, block_size_y, 1);
	HikariTestOwlTonemap_estimateLuminanceImpl << <grid, threads , 0, stream>> > (width, height, input_buffer, luminance_buffer, luminance_log_buffer);
}

void HikariTestOwlTonemap_estimateMaxAndAverage(cudaStream_t  stream, int width, int height, const float* luminance_buffer, const float* luminance_log_buffer, float* p_max_luminance, float* p_ave_luminance)
{
	auto luminance_buffer_ptr     = thrust::device_pointer_cast(luminance_buffer);
	auto luminance_log_buffer_ptr = thrust::device_pointer_cast(luminance_log_buffer);
	auto  max_iter  = thrust::max_element(thrust::cuda::par.on(stream),luminance_buffer_ptr, luminance_buffer_ptr + width * height);
	float max_value = 0.0f;
	cudaMemcpyAsync(&max_value, max_iter.get(), sizeof(float), cudaMemcpyDeviceToHost,stream);
	auto ave_value = expf(thrust::reduce(luminance_log_buffer_ptr, luminance_log_buffer_ptr + width * height) / (width * height));
	printf("value=%f %f\n", ave_value, max_value);
	if (p_ave_luminance) *p_ave_luminance = ave_value;
	if (p_max_luminance) *p_max_luminance = max_value;
}

void HikariTestOwlTonemap_tonemapColorRGBA8(cudaStream_t  stream, int width, int height, const float3* input_buffer, unsigned int* output_buffer, float max_luminance, float ave_luminance, float key_value)
{
	const int block_size_x = 32;
	const int block_size_y = 32;
	const int grid_size_x  = (width  + block_size_x - 1) / block_size_x;
	const int grid_size_y  = (height + block_size_y - 1) / block_size_y;
	
	dim3 grid(grid_size_x, grid_size_y,1);
	dim3 threads(block_size_x, block_size_y, 1);
	HikariTestOwlTonemap_tonemapColorRGBA8Impl<<<grid, threads, 0,stream >>>(width,height,input_buffer,output_buffer, max_luminance,ave_luminance, key_value);
}
