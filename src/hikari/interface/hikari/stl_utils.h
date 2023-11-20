#ifndef HK_STL_UTILS__H
#define HK_STL_UTILS__H

#if defined(__cplusplus) && !defined(__CUDACC__)

template<typename ArrayType>
HK_INLINE std::vector<typename ArrayType::value_type> HKSTLUtils_toSTLArray(ArrayType* array_ptr) {
    std::vector<typename ArrayType::value_type> res = {};
    res.resize(array_ptr->getCount());
    for (auto i = 0;i<res.size();++i){
        res[i] = array_ptr->getValue(i);
    }
    array_ptr->release();
    return res;
}

#endif

#endif
