#pragma once
#include <cuda_runtime.h>
#if defined(__cplusplus)
extern "C" {
#endif
    void HikariTestOwlTonemap_estimateLuminance(
        cudaStream_t  stream,
        int           width,
        int           height,
        const float3* input_buffer,
        float*        luminance_buffer,
        float*        luminance_log_buffer
    );
    void HikariTestOwlTonemap_estimateMaxAndAverage(
        cudaStream_t  stream,
        int           width,
        int           height,
        const float*  luminance_buffer,
        const float*  luminance_log_buffer,
        float*        p_max_luminance,
        float*        p_ave_luminance
    );
    void HikariTestOwlTonemap_tonemapColorRGBA8(
        cudaStream_t  stream,
        int           width        ,
        int           height       ,
        const float3* input_buffer ,
        unsigned int* output_buffer,
        float         max_luminance,
        float         ave_luminance,
        float         key_value
    );

#if defined(__cplusplus)
}
#endif