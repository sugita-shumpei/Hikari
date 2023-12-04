#pragma once

#if defined(__cplusplus)
extern "C" {
#endif
	void calculateLogAverageAndMax(
		int    width,
		int    height,
		float* data_device_ptr,
		float* data_log_device_ptr,
		float* max_value,
		float* log_average_value
	);
#if defined(__cplusplus)
}
#endif