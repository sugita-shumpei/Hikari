#include "kernelCode.h"
#define EPS 0.001f
// ‹ß–T‰æ‘f‚Ì‹P“x‚©‚çÅ‘å’l‚Æ‘Î”’l‚ð‹‚ß‚é
extern "C" __global__ void __kernel__convertToLuminance(
    int   width, int  height   ,
    const float3 *    color3f  ,
    float *           luminance,
    float *           luminance_log)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width && j >= height) {
        return;
    }
    float l     = convertRgbToL(color3f[width * j + i]);
    // ‹P“x‚ð‹‚ß‚é
    luminance[j*width+i]     = l;
    luminance_log[j*width+i] = logf(l+EPS);
}

extern "C" __global__ void __kernel__convertToRGBA8(
    int width,  int      height ,
    const float3       * color3f,
    unsigned int       * color32)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ( i >= width && j >= height){
        return;
    }
    auto in_col     = color3f[width * j + i];
    unsigned int r  = fminf(255,fmaxf(0,int(in_col.x*256.f)));
    unsigned int g  = fminf(255,fmaxf(0,int(in_col.y*256.f)));
    unsigned int b  = fminf(255,fmaxf(0,int(in_col.z*256.f)));
    color32[width*j+i] = (r << 0u) +(g<< 8u)+(b<<16u)+(0xffu<<24u);
}

extern "C" __global__ void __kernel__tonemap(
    int width, int height , 
    float        key_value,
    float   aver_luminance,
    float    max_luminance,
    const float3 * color3f,
    unsigned int*  color32)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width && j >= height) {
        return;
    }
    
    auto  in_col         = color3f[width * j + i];
    float max_luminance_ = key_value* max_luminance / aver_luminance;
    float luminance      = convertRgbToL(in_col);
    float luminance_     = key_value* luminance / aver_luminance;
    float fin_luminance  = luminance_ * (1.0f + (luminance_ / (max_luminance_ * max_luminance_))) / (1.0f + luminance_);
   /* in_col.x *= (key_value / aver_luminance);
    in_col.y *= (key_value / aver_luminance);
    in_col.z *= (key_value / aver_luminance);

    in_col.x = (in_col.x / (1.0f + in_col.x)) * (1.0f + (in_col.x / (max_luminance_ * max_luminance_)));
    in_col.y = (in_col.y / (1.0f + in_col.y)) * (1.0f + (in_col.y / (max_luminance_ * max_luminance_)));
    in_col.z = (in_col.z / (1.0f + in_col.z)) * (1.0f + (in_col.z / (max_luminance_ * max_luminance_)));*/

    in_col.x = (in_col.x / luminance) * fin_luminance;
    in_col.y = (in_col.y / luminance) * fin_luminance;
    in_col.z = (in_col.z / luminance) * fin_luminance;
    //in_col.x = (in_col.x / (1.0f + in_col.x));
    //in_col.y = (in_col.y / (1.0f + in_col.y));
    //in_col.z = (in_col.z / (1.0f + in_col.z));
    //in_col.x = powf(in_col.x, 1.0f / 2.2f);
    //in_col.y = powf(in_col.y, 1.0f / 2.2f);
    //in_col.z = powf(in_col.z, 1.0f / 2.2f);

    unsigned int r = fminf(255, fmaxf(0, int(in_col.x * 256.f)));
    unsigned int g = fminf(255, fmaxf(0, int(in_col.y * 256.f)));
    unsigned int b = fminf(255, fmaxf(0, int(in_col.z * 256.f)));
    color32[width * j + i] = (r << 0u) + (g << 8u) + (b << 16u) + (0xffu << 24u);
}
