#pragma once
#include <tuple>
#include <cmath>
#include <owl/common/math/vec.h>
#include <owl/common/math/constants.h>

namespace hikari {
	namespace test {
		namespace owl {
			namespace testlib {
				struct PinholeCamera {
					auto getUVW() const->std::tuple<::owl::vec3f, ::owl::vec3f, ::owl::vec3f> {
						auto eye = origin;
						auto dir_w = ::owl::normalize(direction);
						auto dir_u = ::owl::normalize(::owl::cross(vup, dir_w));
						auto dir_v = ::owl::normalize(::owl::cross(dir_w, dir_u));
						auto h = std::tan(M_PI * fovy / 360.0f);
						auto w = getAspect() * h;
						dir_u *= h;
						dir_v *= h;
						return std::make_tuple(dir_u, dir_v, dir_w);
					}
					auto getAspect() const -> float { return static_cast<float>(width) / static_cast<float>(height); }
					void processPressKeyW(float delta) {
						origin += delta * direction * speed.y;
					}
					void processPressKeyA(float delta) {
						auto dir_w = ::owl::normalize(direction);
						auto dir_u = ::owl::normalize(::owl::cross(vup, dir_w));
						origin -= delta * dir_u * speed.x;
					}
					void processPressKeyS(float delta) {
						origin -= delta * direction * speed.y;
					}
					void processPressKeyD(float delta) {
						auto dir_w = ::owl::normalize(direction);
						auto dir_u = ::owl::normalize(::owl::cross(vup, dir_w));
						origin += delta * dir_u * speed.x;
					}
					void processPressKeyUp(float delta) {
						// A
						// |
						// ----->
						auto len_w = ::owl::length(direction);
						auto dir_w = ::owl::normalize(direction);
						auto dir_u = ::owl::normalize(::owl::cross(vup, dir_w));
						auto dir_v = ::owl::normalize(::owl::cross(dir_w, dir_u));

						auto theta = M_PI * delta / 180.0f;
						auto cos_f = std::cosf(theta);
						auto sin_f = std::sinf(theta);

						direction = len_w * (dir_w * cos_f + dir_v * sin_f);
					}
					void processPressKeyDown(float delta) {
						auto len_w = ::owl::length(direction);
						auto dir_w = ::owl::normalize(direction);
						auto dir_u = ::owl::normalize(::owl::cross(vup, dir_w));
						auto dir_v = ::owl::normalize(::owl::cross(dir_w, dir_u));

						auto theta = M_PI * delta / 180.0f;
						auto cos_f = std::cosf(theta);
						auto sin_f = std::sinf(theta);

						direction = len_w * (dir_w * cos_f - dir_v * sin_f);
					}
					void processPressKeyLeft(float delta) {
						// A
						// |
						// ----->
						auto len_w = ::owl::length(direction);
						auto dir_w = ::owl::normalize(direction);
						auto dir_u = ::owl::normalize(::owl::cross(vup, dir_w));
						auto dir_v = ::owl::normalize(::owl::cross(dir_w, dir_u));

						auto theta = M_PI * delta / 180.0f;
						auto cos_f = std::cosf(theta);
						auto sin_f = std::sinf(theta);

						direction = len_w * (dir_w * cos_f - dir_u * sin_f);
					}
					void processPressKeyRight(float delta) {
						auto len_w = ::owl::length(direction);
						auto dir_w = ::owl::normalize(direction);
						auto dir_u = ::owl::normalize(::owl::cross(vup, dir_w));
						auto dir_v = ::owl::normalize(::owl::cross(dir_w, dir_u));

						auto theta = M_PI * delta / 180.0f;
						auto cos_f = std::cosf(theta);
						auto sin_f = std::sinf(theta);

						direction = len_w * (dir_w * cos_f + dir_u * sin_f);
					}
					::owl::vec3f origin = ::owl::vec3f(0.0f, 0.0f, -1.0f);
					::owl::vec3f direction = ::owl::vec3f(0.0f, 0.0f, 1.0f);
					::owl::vec3f vup = ::owl::vec3f(0.0f, 1.0f, 0.0f);
					::owl::vec2f speed = ::owl::vec2f(1.0f, 1.0f);
					float fovy   = 30.0f;
					int   width  = 1024;
					int   height = 1024;
				};
			}
		}
	}
}
