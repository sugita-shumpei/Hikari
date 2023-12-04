#include "thrustCode.h"
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
extern "C" {
	void calculateLogAverageAndMax(
		int    width            ,
		int    height           ,
		float* data_device_ptr  ,
		float* data_log_device_ptr,
		float* max_value        ,
		float* log_average_value)
	{
		auto data     = thrust::device_pointer_cast(data_device_ptr);
		auto log_data = thrust::device_pointer_cast(data_log_device_ptr);
		auto iter     = thrust::max_element(data, data + width * height);
		float value   = 0.0f;
		cudaMemcpy(&value,iter.get(), sizeof(float), cudaMemcpyDeviceToHost);
		auto average  = expf(thrust::reduce(log_data, log_data + width * height)/(width*height));
		printf("%f %f\n", value, average);
		if(max_value) *max_value = value;
		if (log_average_value) *log_average_value = average;
	}
}