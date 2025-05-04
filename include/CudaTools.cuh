#pragma once
#include <iostream>
#include <cuda_runtime.h>

inline void getLastCudaError(const char* msg, const char* file, int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << "\n"
            << "  -> Error: " << cudaGetErrorString(err) << "\n"
            << "  -> File: " << file << "\n"
            << "  -> Line: " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}


#define CHECK_CUDA_ERROR(msg) getLastCudaError(msg, __FILE__, __LINE__)