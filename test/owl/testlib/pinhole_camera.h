#pragma once
#include <tuple>
#include <cmath>
#include <glm/glm.hpp>

namespace hikari {
	namespace test {
		namespace owl {
			namespace testlib {
				struct PinholeCamera {
					auto getUVW() const->std::tuple<::glm::vec3, ::glm::vec3, ::glm::vec3> {
						auto eye = origin;
						auto dir_w = ::glm::normalize(direction);
						auto dir_u = ::glm::normalize(::glm::cross(vup, dir_w));
						auto dir_v = ::glm::normalize(::glm::cross(dir_w, dir_u));
						auto h = std::tan(M_PI * fovy / 360.0f);
						auto w = getAspect() * h;
						dir_u *= h;
						dir_v *= h;
						return std::make_tuple(dir_u, dir_v, dir_w);
					}
					auto getAspect() const -> float { return static_cast<float>(width) / static_cast<float>(height); }
					void processPressKeyW(float delta) {
						origin += delta * direction * speed.z;
					}
					void processPressKeyA(float delta) {
						auto dir_w = ::glm::normalize(direction);
						auto dir_u = ::glm::normalize(::glm::cross(vup, dir_w));
						origin -= delta * dir_u * speed.x;
					}
					void processPressKeyS(float delta) {
						origin -= delta * direction * speed.z;
					}
					void processPressKeyD(float delta) {
						auto dir_w = ::glm::normalize(direction);
						auto dir_u = ::glm::normalize(::glm::cross(vup, dir_w));
						origin += delta * dir_u * speed.x;
					}
					void processPressKeyUp(float delta) {
						// A
						// |
						// ----->
						auto len_w = ::glm::length(direction);
						auto dir_w = ::glm::normalize(direction);
						auto dir_u = ::glm::normalize(::glm::cross(vup, dir_w));
						auto dir_v = ::glm::normalize(::glm::cross(dir_w, dir_u));

						auto theta = M_PI * delta / 180.0f;
						auto cos_f = std::cosf(theta);
						auto sin_f = std::sinf(theta);

						direction = len_w * (dir_w * cos_f + dir_v * sin_f);
					}
					void processPressKeyDown(float delta) {
						auto len_w = ::glm::length(direction);
						auto dir_w = ::glm::normalize(direction);
						auto dir_u = ::glm::normalize(::glm::cross(vup, dir_w));
						auto dir_v = ::glm::normalize(::glm::cross(dir_w, dir_u));

						auto theta = M_PI * delta / 180.0f;
						auto cos_f = std::cosf(theta);
						auto sin_f = std::sinf(theta);

						direction = len_w * (dir_w * cos_f - dir_v * sin_f);
					}
					void processPressKeyLeft(float delta) {
						// A
						// |
						// ----->
						auto len_w = ::glm::length(direction);
						auto dir_w = ::glm::normalize(direction);
						auto dir_u = ::glm::normalize(::glm::cross(vup, dir_w));
						auto dir_v = ::glm::normalize(::glm::cross(dir_w, dir_u));

						auto theta = M_PI * delta / 180.0f;
						auto cos_f = std::cosf(theta);
						auto sin_f = std::sinf(theta);

						direction = len_w * (dir_w * cos_f - dir_u * sin_f);
					}
					void processPressKeyRight(float delta) {
						auto len_w = ::glm::length(direction);
						auto dir_w = ::glm::normalize(direction);
						auto dir_u = ::glm::normalize(::glm::cross(vup, dir_w));
						auto dir_v = ::glm::normalize(::glm::cross(dir_w, dir_u));

						auto theta = M_PI * delta / 180.0f;
						auto cos_f = std::cosf(theta);
						auto sin_f = std::sinf(theta);

						direction = len_w * (dir_w * cos_f + dir_u * sin_f);
					}
					void processMouseScrollY(float delta) {
						auto dir_w = ::glm::normalize(direction);
						auto dir_u = ::glm::normalize(::glm::cross(vup, dir_w));
						auto dir_v = ::glm::normalize(::glm::cross(dir_w, dir_u));
						origin += delta * dir_v * speed.y;
					}
					::glm::vec3 origin    = ::glm::vec3(0.0f, 0.0f, -1.0f);
					::glm::vec3 direction = ::glm::vec3(0.0f, 0.0f, 1.0f);
					::glm::vec3 vup       = ::glm::vec3(0.0f, 1.0f, 0.0f);
					::glm::vec3 speed     = ::glm::vec3(1.0f, 1.0f,1.0f);
					float fovy   = 30.0f;
					int   width  = 1024;
					int   height = 1024;
				};
                        }
		}
	}
}
