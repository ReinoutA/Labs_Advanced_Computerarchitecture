#include <iostream>
#include <cstdlib>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>

#define N 32768
#define NUM_MEASUREMENTS 100
#define BLOCK_SIZE 256

// Kernelfuncties voor de reductie-operaties
__device__ void reductionOperation(int* arr, int j, unsigned int i, int operation) {
    for (; i % j == 0 && j < N; j *= 2) {
        for (int k = 0; i + k * gridDim.x * blockDim.x + j < N; k++) {
            int index = i + k * gridDim.x * blockDim.x;
            switch (operation) {
            case 0: // Max
                if (arr[2 * index] < arr[2 * index + j]) arr[2 * index] = arr[2 * index + j];
                break;
            case 1: // Min
                if (arr[2 * index] > arr[2 * index + j]) arr[2 * index] = arr[2 * index + j];
                break;
            case 2: // Sum
                arr[2 * index] += arr[2 * index + j];
                break;
            case 3: // Multiply
                arr[2 * index] *= arr[2 * index + j];
                break;
            }
        }
        __syncthreads();
    }
}


__global__ void reductionMax(int* arr) {
    int j = 1;
    unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x);
    reductionOperation(arr, j, i, 0);
}

__global__ void reductionMin(int* arr) {
    int j = 1;
    unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x);
    reductionOperation(arr, j, i, 1);
}

__global__ void reductionSum(int* arr) {
    int j = 1;
    unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x);
    reductionOperation(arr, j, i, 2);
}

__global__ void reductionMultiply(int* arr) {
    int j = 1;
    unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x);
    reductionOperation(arr, j, i, 3);
}

int* generate(int size) {
    int* arr = new int[size];
    srand(time(nullptr));

    for (int i = 0; i < size; i++) {
        arr[i] = (rand() % 10) + 1;
    }

    return arr;
}

float measureExecutionTime(void (*reductionFunc)(int*), int* data_cuda) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reductionFunc << <1, 48 >> > (data_cuda);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}

float async_reduction(int arraySize) {
    const auto start_time_async = std::chrono::steady_clock::now();
    int* sum_array = generate(arraySize);
    int* sum_cuda_data;
    cudaMalloc(&sum_cuda_data, arraySize * sizeof(int));
    cudaMemcpy(sum_cuda_data, sum_array, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    int* product_array = generate(arraySize);
    int* prod_cuda_data;
    cudaMalloc(&prod_cuda_data, arraySize * sizeof(int));
    cudaMemcpy(prod_cuda_data, product_array, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    int* minimum_array = generate(arraySize);
    int* min_cuda_data;
    cudaMalloc(&min_cuda_data, arraySize * sizeof(int));
    cudaMemcpy(min_cuda_data, minimum_array, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    int* maximum_array = generate(arraySize);
    int* max_cuda_data;
    cudaMalloc(&max_cuda_data, arraySize * sizeof(int));
    cudaMemcpy(max_cuda_data, maximum_array, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    float sum_time_async = measureExecutionTime(reductionSum, sum_cuda_data);
    float prod_time_async = measureExecutionTime(reductionMultiply, prod_cuda_data);
    float min_time_async = measureExecutionTime(reductionMin, min_cuda_data);
    float max_time_async = measureExecutionTime(reductionMax, max_cuda_data);
    cudaFree(sum_cuda_data);
    cudaFree(prod_cuda_data);
    cudaFree(min_cuda_data);
    cudaFree(max_cuda_data);
    const auto end_time_async = std::chrono::steady_clock::now();
    const std::chrono::duration<double> seconds_async = end_time_async - start_time_async;
    delete[] sum_array;
    delete[] product_array;
    delete[] minimum_array;
    delete[] maximum_array;
    return seconds_async.count();
}

float sync_reduction(int arraySize) {
    const auto start_time_sync = std::chrono::steady_clock::now();
    int* sum_array = generate(arraySize);
    int* sum_cuda_data;
    cudaMalloc(&sum_cuda_data, arraySize * sizeof(int));
    cudaMemcpy(sum_cuda_data, sum_array, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    float sum_time_sync = measureExecutionTime(reductionSum, sum_cuda_data);
    cudaFree(sum_cuda_data);
    int* product_array = generate(arraySize);
    int* prod_cuda_data;
    cudaMalloc(&prod_cuda_data, arraySize * sizeof(int));
    cudaMemcpy(prod_cuda_data, product_array, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    float prod_time_sync = measureExecutionTime(reductionMultiply, prod_cuda_data);
    cudaFree(prod_cuda_data);
    int* minimum_array = generate(arraySize);
    int* min_cuda_data;
    cudaMalloc(&min_cuda_data, arraySize * sizeof(int));
    cudaMemcpy(min_cuda_data, minimum_array, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    float min_time_sync = measureExecutionTime(reductionMin, min_cuda_data);
    cudaFree(min_cuda_data);
    int* maximum_array = generate(arraySize);
    int* max_cuda_data;
    cudaMalloc(&max_cuda_data, arraySize * sizeof(int));
    cudaMemcpy(max_cuda_data, maximum_array, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    float max_time_sync = measureExecutionTime(reductionMax, max_cuda_data);
    cudaFree(max_cuda_data);
    const auto end_time_sync = std::chrono::steady_clock::now();
    const std::chrono::duration<double> seconds_sync = end_time_sync - start_time_sync;
    delete[] sum_array;
    delete[] product_array;
    delete[] minimum_array;
    delete[] maximum_array;
    return seconds_sync.count();
}

float sync = 0;
float async = 0;

int main() {
    std::ofstream outputFile("results.csv");
    if (!outputFile.is_open()) {
        std::cerr << "Error: Could not open the output file." << std::endl;
        return 1;
    }

    outputFile << "ArraySize,Sequential,Parallel" << std::endl;
    bool first = true;

    for (int arraySize = 4; arraySize <= N; arraySize += 256) {
        std::cout << arraySize << std::endl;
        sync = 0;
        async = 0;


        for (int m = 0; m < NUM_MEASUREMENTS; m++)
            sync += sync_reduction(arraySize) * 1000;
        

        for (int m = 0; m < NUM_MEASUREMENTS; m++)
            async += async_reduction(arraySize) * 1000;

        

        if (!first) {
            outputFile << arraySize << ",";
            outputFile << sync / NUM_MEASUREMENTS << ",";
            outputFile << async / NUM_MEASUREMENTS << std::endl;
        }
        else first = false;
    }

    outputFile.close();
    return 0;
}