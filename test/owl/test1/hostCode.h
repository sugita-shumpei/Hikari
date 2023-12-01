#pragma once
#include <tuple>
#include <cmath>
#include <owl/common/math/vec.h>
#include <owl/common/math/constants.h>

struct PinholeCamera {
	auto getUVW() const->std::tuple<owl::vec3f, owl::vec3f, owl::vec3f> {
		auto eye = origin;
		auto dir_w = owl::normalize(direction);
		auto dir_u = owl::normalize(owl::cross(vup, dir_w));
		auto dir_v = owl::normalize(owl::cross(dir_w, dir_u));
		auto h = std::tan(M_PI * fovy / 180.0f);
		auto w = getAspect() * h;
		dir_u *= h;
		dir_v *= h;
		return std::make_tuple(dir_u, dir_v, dir_w);
	}
	auto getAspect() const -> float { return static_cast<float>(width) / static_cast<float>(height); }
	void processPressKeyW(float delta) {
		origin += delta * direction;
	}
	void processPressKeyA(float delta) {
		auto dir_w = owl::normalize(direction);
		auto dir_u = owl::normalize(owl::cross(vup, dir_w));
		origin -= delta * dir_u;
	}
	void processPressKeyS(float delta) {
		origin -= delta * direction;
	}
	void processPressKeyD(float delta) {
		auto dir_w = owl::normalize(direction);
		auto dir_u = owl::normalize(owl::cross(vup, dir_w));
		origin += delta * dir_u;
	}
	owl::vec3f origin = owl::vec3f(0.0f, 0.0f, -1.0f);
	owl::vec3f direction = owl::vec3f(0.0f, 0.0f, 1.0f);
	owl::vec3f vup = owl::vec3f(0.0f, 1.0f, 0.0f);
	float fovy   = 30.0f;
	int   width  = 1024;
	int   height = 1024;
};