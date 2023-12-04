#include "tonemap.h"
#include <cuda/tonemap_impl.h>

hikari::test::owl::testlib::Tonemap::Tonemap(int width, int height, float key_value) :
	m_width{ width },
	m_height{ height },
	m_key_value{ key_value },
	m_ave_luminance{ 0.0f },
	m_max_luminance{ 0.0f },
	m_luminance_buffer{ nullptr },
	m_luminance_log_buffer{ nullptr }
{
}

hikari::test::owl::testlib::Tonemap::~Tonemap()
{
}

void hikari::test::owl::testlib::Tonemap::init()
{

	int width = m_width; int height = m_height;
	m_width = 0; m_height = 0;
	resize(width, height);
}

void hikari::test::owl::testlib::Tonemap::free()
{
	if (m_luminance_buffer)     cuMemFree(reinterpret_cast<CUdeviceptr>(m_luminance_buffer));
	if (m_luminance_log_buffer) cuMemFree(reinterpret_cast<CUdeviceptr>(m_luminance_log_buffer));
	m_luminance_buffer     = nullptr;
	m_luminance_log_buffer = nullptr;
}

void hikari::test::owl::testlib::Tonemap::launch(CUstream      stream, const float3* input_buffer, unsigned int* output_buffer, bool skip_estimation)
{
	if (!skip_estimation) {
		estimateLuminance(stream,input_buffer);
		estimateMaxAndAverage(stream);
	}
	tonemapColorRGBA8(stream, input_buffer, output_buffer);
}

void hikari::test::owl::testlib::Tonemap::resize(int width, int height)
{
	if (m_width == width && m_height == height) { return; }
	if (m_luminance_buffer)     cuMemFree(reinterpret_cast<CUdeviceptr>(m_luminance_buffer));
	if (m_luminance_log_buffer) cuMemFree(reinterpret_cast<CUdeviceptr>(m_luminance_log_buffer));
	CUdeviceptr tmp_luminance_buffer;
	CUdeviceptr tmp_luminance_log_buffer;
	cuMemAlloc(&tmp_luminance_buffer    , width * height * sizeof(float));
	cuMemAlloc(&tmp_luminance_log_buffer, width * height * sizeof(float));
	m_luminance_buffer     = reinterpret_cast<float*>(tmp_luminance_buffer);
	m_luminance_log_buffer = reinterpret_cast<float*>(tmp_luminance_log_buffer);
	m_width = width;
	m_height = height;
}

void hikari::test::owl::testlib::Tonemap::estimateLuminance(CUstream stream, const float3* input_buffer)
{
	HikariTestOwlTonemap_estimateLuminance(stream,m_width, m_height, input_buffer, m_luminance_buffer, m_luminance_log_buffer);
}

void hikari::test::owl::testlib::Tonemap::estimateMaxAndAverage(CUstream stream)
{
	HikariTestOwlTonemap_estimateMaxAndAverage(stream, m_width, m_height, m_luminance_buffer, m_luminance_log_buffer, &m_max_luminance, &m_ave_luminance);
}

void hikari::test::owl::testlib::Tonemap::tonemapColorRGBA8(CUstream stream, const float3* input_buffer, unsigned int* result_buffer)
{
	HikariTestOwlTonemap_tonemapColorRGBA8(stream, m_width, m_height, input_buffer, result_buffer, m_max_luminance, m_ave_luminance,m_key_value);
}
