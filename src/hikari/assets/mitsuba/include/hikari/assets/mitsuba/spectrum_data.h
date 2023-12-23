#pragma once
#include <hikari/core/data_type.h>
#include <algorithm>
namespace hikari
{
    struct MitsubaSpectrumData
    {
        Bool load(const String &filename);
        static auto d65()       -> MitsubaSpectrumData;
        static auto cie1931_x() -> MitsubaSpectrumData;
        static auto cie1931_y() -> MitsubaSpectrumData;
        static auto cie1931_z() -> MitsubaSpectrumData;

        auto getValue(F32 wavelength)const->F32 {
          auto iter = std::ranges::upper_bound(wavelengths, wavelength);
          // 先頭の場合
          if (iter == std::begin(wavelengths)) {
            return 0.0f;
          }
          // 末端の場合
          if (iter == std::end(wavelengths)) {
            return 0.0f;
          }
          auto idx = std::distance(std::begin(wavelengths), iter) - 1;
          auto v0  = values[idx + 0];
          auto v1  = values[idx + 1];
          auto w0  = wavelengths[idx + 0];
          auto w1  = wavelengths[idx + 1];
          return ((v1 - v0) / (w1 - w0)) * (wavelength - w0) + v0;
        }
        auto getAverageValue(F32 beg_len, F32 end_len) const->F32 {
          auto beg_iter = std::ranges::upper_bound(wavelengths, beg_len);
          auto end_iter = std::ranges::upper_bound(wavelengths, end_len);
          {
            // 区間積分を行う
            auto beg_range = std::distance(std::begin(wavelengths), beg_iter);
            auto end_range = std::distance(std::begin(wavelengths), end_iter);
            auto max_range = wavelengths.size();

            if (beg_range == 0) {
              beg_len = wavelengths.front();
            }
            else if (beg_range == max_range) {
              beg_len   = wavelengths.back();
              beg_range = max_range - 1;
            }
            if (end_range == 0) {
              end_len = wavelengths.front();
            }
            else if (end_range == max_range) {
              end_len = wavelengths.back();
              end_range = max_range - 1;
            }
            if (beg_len >= end_len) { return 0.0f; }

            auto res = 0.0f;
            if (beg_range > 1) {
              auto v0 = values[beg_range - 1];
              auto v1 = values[beg_range + 0];
              auto w0 = wavelengths[beg_range - 1];
              auto w1 = wavelengths[beg_range + 0];
              auto wt = beg_len;
              auto t  = (v1-v0)/(w1 - w0);
              auto vt = t * (wt -w0) + v0;
              auto av = (vt + v1) * 0.5f;
              auto dw = w1 - wt;
              res    += (av * dw);
            }
            if (end_range> 1){
              for (size_t i = beg_range; i < end_range - 1; ++i) {
                auto dw = wavelengths[i + 1] - wavelengths[i];
                auto av = (values[i + 0] + values[i + 1]) * 0.5f;
                res += av * dw;
              }
            }
            if (end_range > 1) {
              auto v0 = values[end_range - 1];
              auto v1 = values[end_range + 0];
              auto w0 = wavelengths[end_range - 1];
              auto w1 = wavelengths[end_range + 0];
              auto wt = end_len;
              auto t  = (v1 - v0) / (w1 - w0);
              auto vt = t * (end_len - w0) + v0;
              auto av = (vt + v0) * 0.5f;
              auto dw = wt - w0;
              res    += av * dw;
            }
            res /= (end_len - beg_len);
            return res;
          }
        }

        std::vector<F32> wavelengths;
        std::vector<F32> values;
    };
}
