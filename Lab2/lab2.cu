#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>

// Function to find the max and its index using a nested for loop on the CPU
int findMaxIndexCPU(const int* arr, int size, int& maxIndex) {
    int maxVal = arr[0];
    maxIndex = 0;
    for (int i = 1; i < size; ++i) {
        if (arr[i] > maxVal) {
            maxVal = arr[i];
            maxIndex = i;
        }
    }
    return maxVal;
}

// GPU kernel using atomic operation to find the max and its index
__global__ void findMaxAtomic(int* arr, int* result, int* index, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        int val = arr[tid];

        // Perform an atomicMax operation to update the max result
        atomicMax(result, val);

        // Check if the current value is the new maximum, if true, update the index
        if (val == *result) {
            atomicExch(index, tid);
        }
    }
}

// GPU kernel using global memory for reduction to find the max and its index
__global__ void findMaxReduction(const int* arr, int* result, int* index, int size) {
    extern __shared__ int sharedMem[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int localMax = (i < size) ? arr[i] : INT_MIN;
    int localIndex = (i < size) ? i : -1;

    // Perform reduction in shared memory
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride && i + stride < size) {
            int otherVal = arr[i + stride];
            if (otherVal > localMax) {
                localMax = otherVal;
                localIndex = i + stride;
            }
        }
    }

    // Store the reduced value and index in shared memory
    if (tid == 0) {
        sharedMem[blockIdx.x * 2] = localMax;
        sharedMem[blockIdx.x * 2 + 1] = localIndex;
    }

    __syncthreads();

    // Perform a final reduction in shared memory to find the overall max
    if (blockIdx.x == 0 && tid == 0) {
        localMax = sharedMem[0];
        localIndex = sharedMem[1];
        for (int j = 1; j < gridDim.x; ++j) {
            int otherVal = sharedMem[j * 2];
            int otherIndex = sharedMem[j * 2 + 1];
            if (otherVal > localMax) {
                localMax = otherVal;
                localIndex = otherIndex;
            }
        }

        // Write the result back to global memory
        atomicMax(result, localMax);
        atomicExch(index, localIndex);
    }
}

/* SOURCE: https://github.com/danielll-w/cuda-maximum-reduction-kernel/blob/main/max_reduction.cu

__global__ void findMaxReduction(int* data, int* result, int* index, int data_size) {
    extern __shared__ int sharedWarpMax[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int d = (tid < data_size) ? data[tid] : INT_MIN;

    int temp_comparison;
    int mask;
    int i;

    for (mask = 1, i = 0; i < 6; mask *= 2, ++i) {
        temp_comparison = __shfl_xor_sync(0xFFFFFFFF, d, mask);
        d = d > temp_comparison ? d : temp_comparison;
    }
 
    if (threadIdx.x % 32 == 0) {
        sharedWarpMax[threadIdx.x / warpSize] = d;
    }

    __syncthreads();

    d = (threadIdx.x < blockDim.x / warpSize) ? sharedWarpMax[threadIdx.x] : INT_MIN;

    for (mask = 1, i = 0; i < 6; mask *= 2, ++i) {
        temp_comparison = __shfl_xor_sync(0xFFFFFFFF, d, mask);
        d = d > temp_comparison ? d : temp_comparison;
    }

    if (threadIdx.x == 0) {
        result[blockIdx.x] = d;
    }

    if (tid == 0) {
        int maxVal = INT_MIN;
        int maxIdx = 0;

        for (int i = 0; i < gridDim.x; ++i) {
            if (result[i] > maxVal) {
                maxVal = result[i];
                maxIdx = i * blockDim.x + threadIdx.x;
            }
        }

        *result = maxVal;
        *index = maxIdx;
    }
}

*/
int main() {
    const int maxArraySize = 1024;  // Maximum array size
    const int stepSize = 1;         // Step size for varying array size

    std::ofstream outputFile("execution_times.csv");
    outputFile << "Array Size,CPU,GPU_A,GPU_R\n";

    for (int arraySize = stepSize; arraySize <= maxArraySize; arraySize += stepSize) {
        int* h_array = new int[arraySize];
        int* d_array, *d_result, *d_index;

        std::srand(static_cast<unsigned>(std::time(nullptr)));
        for (int i = 0; i < arraySize; ++i) {
            h_array[i] = std::rand() % 1000001;  // Random number between 0 and 1 million
        }

        // CPU max detection
        int maxIndexCPU;
        auto startCPU = std::chrono::high_resolution_clock::now();
        int maxCPU = findMaxIndexCPU(h_array, arraySize, maxIndexCPU);
        auto endCPU = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpuTime = endCPU - startCPU;

        std::cout << "Array Size: " << arraySize << std::endl;
        std::cout << "CPU - Max: " << maxCPU << ", Index: " << maxIndexCPU << ", Time: " << cpuTime.count() << " seconds" << std::endl;

        // GPU max detection using atomic operation
        cudaMalloc((void**)&d_array, arraySize * sizeof(int));
        cudaMemcpy(d_array, h_array, arraySize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_result, sizeof(int));
        cudaMalloc((void**)&d_index, sizeof(int));
        cudaMemset(d_result, 0, sizeof(int));
        cudaMemset(d_index, 0, sizeof(int));

        dim3 blockDim(256);
        dim3 gridDim((arraySize + blockDim.x - 1) / blockDim.x);

        cudaEvent_t startAtomic, endAtomic;
        cudaEventCreate(&startAtomic);
        cudaEventCreate(&endAtomic);

        cudaEventRecord(startAtomic);
        findMaxAtomic<<<gridDim, blockDim>>>(d_array, d_result, d_index, arraySize);
        cudaEventRecord(endAtomic);
        cudaEventSynchronize(endAtomic);

        float atomicTime;
        cudaEventElapsedTime(&atomicTime, startAtomic, endAtomic);

        int maxAtomic, maxIndexAtomic;
        cudaMemcpy(&maxAtomic, d_result, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&maxIndexAtomic, d_index, sizeof(int), cudaMemcpyDeviceToHost);

        std::cout << "GPU Atomic - Max: " << maxAtomic << ", Index: " << maxIndexAtomic << ", Time: " << atomicTime / 1000.0 << " seconds" << std::endl;

        // GPU max detection using reduction
        cudaMemset(d_result, 0, sizeof(int));
        cudaMemset(d_index, 0, sizeof(int));

        cudaEvent_t startReduction, endReduction;
        cudaEventCreate(&startReduction);
        cudaEventCreate(&endReduction);

        cudaEventRecord(startReduction);
        findMaxReduction<<<gridDim, blockDim, blockDim.x * sizeof(int)>>>(d_array, d_result, d_index, arraySize);
        cudaEventRecord(endReduction);
        cudaEventSynchronize(endReduction);

        float reductionTime;
        cudaEventElapsedTime(&reductionTime, startReduction, endReduction);

        int maxReduction, maxIndexReduction;
        cudaMemcpy(&maxReduction, d_result, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&maxIndexReduction, d_index, sizeof(int), cudaMemcpyDeviceToHost);

        std::cout << "GPU Reduction - Max: " << maxReduction << ", Index: " << maxIndexReduction << ", Time: " << reductionTime / 1000.0 << " seconds" << std::endl;

        // Clean up
        delete[] h_array;
        cudaFree(d_array);
        cudaFree(d_result);
        cudaFree(d_index);

        // Write execution times to CSV file
        outputFile << arraySize << ", " << cpuTime.count() << ", " << atomicTime / 1000.0 << ", " << reductionTime / 1000.0 << "\n";
    }

    outputFile.close();

    return 0;
}
