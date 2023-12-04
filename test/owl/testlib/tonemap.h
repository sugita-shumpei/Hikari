#pragma once
#include <cuda.h>
#include <vector_types.h>
#include "common.h"
namespace hikari {
	namespace test {
		namespace owl {
			namespace testlib {
				struct Tonemap {
                    Tonemap(int width, int height, float key_value, TonemapType type = TonemapType::eReinhard);
                    ~Tonemap();

                    void init();
                    void free();
                    void launch(
                        CUstream      stream,
                        const float3* input_buffer,
                        unsigned int* output_buffer,
                        bool          skip_estimation = false
                    );
                    void resize(int width, int height);
                    auto getWidth() const { return m_width; }
                    auto getHeight() const { return m_height; }
                    auto getKeyValue()const { return m_key_value; }
                    void setKeyValue(float v) { m_key_value = v; }
                    auto getType()const { return m_type; }
                    void setType(TonemapType type) { m_type = type;}
                    auto getAveLuminance() const { return m_ave_luminance; }
                    auto getMaxLuminance() const { return m_max_luminance; }
                    auto getLuminanceBuffer() const { return m_luminance_buffer; }
                    auto getLuminanceLogBuffer() const { return m_luminance_buffer; }
                private:
                    void estimateLuminance(CUstream stream,const float3* input_buffer);
                    void estimateMaxAndAverage(CUstream stream);
                    void tonemapColorRGBA8(CUstream stream, const float3* input_buffer, unsigned int* result_buffer);
                    TonemapType m_type;
                    int      m_width ;
                    int      m_height;
                    float    m_key_value;
                    float    m_ave_luminance;
                    float    m_max_luminance ;
                    float*   m_luminance_buffer;
                    float*   m_luminance_log_buffer;
                };
            }
        }
    }
}
