#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <chrono>

#define X_DIM 64*64                  
#define Y_DIM 8      
#define LOWER_BOUND -1                               
#define MAX_THREADS_PER_BLOCK 1024                         
#define NUM_BLOCKS ((X_DIM / MAX_THREADS_PER_BLOCK) + 1)

        
// Reduction within each block, gets passed to second kernel
__device__ int blockVals[Y_DIM][NUM_BLOCKS];       
__device__ int blockIndices[Y_DIM][NUM_BLOCKS];    


__global__ void reduceBlockLevel(const float *data, const int size, const int ySize){
    __shared__ int vals[MAX_THREADS_PER_BLOCK];
    __shared__ int indices[MAX_THREADS_PER_BLOCK];

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = blockIdx.y;
    int localMax = LOWER_BOUND;
    int localIndex = -1;

    while (i < size){
        int val = static_cast<int>(data[tid * size + i]);
        if (val > localMax){
            localIndex = i;
            localMax = val;
        }
        i += blockDim.x * gridDim.x;
    }

    vals[threadIdx.x] = localMax;
    indices[threadIdx.x] = localIndex;

    __syncthreads();

    for (int i = (MAX_THREADS_PER_BLOCK >> 1); i > 0; i >>= 1){
        if (i > threadIdx.x)
            if (vals[threadIdx.x] < vals[threadIdx.x + i]){
                indices[threadIdx.x] = indices[threadIdx.x + i];
                vals[threadIdx.x] = vals[threadIdx.x + i];
            }
        __syncthreads();
    }

    if (!threadIdx.x){
        blockIndices[blockIdx.y][blockIdx.x] = indices[0];
        blockVals[blockIdx.y][blockIdx.x] = vals[0];
        __syncthreads();
    }
}

__global__ void reduceFinal(int *index, int *result){
    __shared__ int vals[MAX_THREADS_PER_BLOCK];
    __shared__ int indices[MAX_THREADS_PER_BLOCK];

    int i = threadIdx.x;
    int tid = blockIdx.y;
    int localMax = LOWER_BOUND;
    int localIndex = -1;

    while (i < NUM_BLOCKS) {
        int val = blockVals[tid][i];
        if (val > localMax) {
            localMax = val;
            localIndex = blockIndices[tid][i];
        }
        i += blockDim.x;
    }

    i = threadIdx.x;
    vals[i] = localMax;
    indices[i] = localIndex;
    __syncthreads();

    for (int j = (MAX_THREADS_PER_BLOCK >> 1); j > 0; j >>= 1) {
        if (i < j)
            if (vals[i] < vals[i + j]) {
                vals[i] = vals[i + j];
                indices[i] = indices[i + j];
            }
        __syncthreads();
    }

    if (!threadIdx.x)
    {
        index[tid] = indices[0];
        result[tid] = vals[0];
    }
}

__global__ void reduceGlobal(const float *data, const int width, const int height, int *index, int *result) {
    __shared__ int vals[MAX_THREADS_PER_BLOCK];
    __shared__ int indices[MAX_THREADS_PER_BLOCK];

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int localMax = LOWER_BOUND;
    int localIndex = -1;

    while (i < width * height) {
        int tid = i / width;
        int columnIndex = i % width;
        int val = static_cast<int>(data[i]);
        if (val > localMax) {
            localMax = val;
            localIndex = columnIndex + tid * width;
        }
        i += blockDim.x * gridDim.x;
    }

    vals[threadIdx.x] = localMax;
    indices[threadIdx.x] = localIndex;

    __syncthreads();

    for (int i = (MAX_THREADS_PER_BLOCK >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i)
            if (vals[threadIdx.x] < vals[threadIdx.x + i]) {
                vals[threadIdx.x] = vals[threadIdx.x + i];
                indices[threadIdx.x] = indices[threadIdx.x + i];
            }
        __syncthreads();
    }

    if (!threadIdx.x) {
        index[blockIdx.x] = indices[0];
        result[blockIdx.x] = vals[0];
    }
}

// Naive CPU Approach
void naiveCPU(const float *data, const int width, const int height, int *index, int *result) {
    auto start = std::chrono::high_resolution_clock::now();

    int globalMax = LOWER_BOUND;
    int globalIndex = -1;

    for (int idy = 0; idy < height; ++idy) {
        for (int columnIndex = 0; columnIndex < width; ++columnIndex) {
            int idx = columnIndex + idy * width;
            int val = static_cast<int>(data[idx]);
            if (val > globalMax) {
                globalMax = val;
                globalIndex = idx;
            }
        }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    printf("Naïeve CPU-benadering uitvoeringstijd: %fms\n", duration.count() / 1000.0);

    *index = globalIndex;
    *result = globalMax;
}

__global__ void atomicMaxKernel(const float *data, const int size, const int ySize, int *index, int *result)
{
    __shared__ int maxVal;
    __shared__ int maxIdx;

    if (threadIdx.x == 0) {
        maxVal = LOWER_BOUND;
        maxIdx = -1;
    }

    __syncthreads();

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    while (idx < size * ySize) {
        int idy = idx / size;
        int columnIndex = idx % size;
        int val = static_cast<int>(data[idx]);
        
        atomicMax(&maxVal, val);
        
        if (val == maxVal) {
            atomicMax(&maxIdx, columnIndex + idy * size);
        }
        idx += blockDim.x * gridDim.x;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        index[blockIdx.x] = maxIdx;
        result[blockIdx.x] = maxVal;
    }
}


int main() {
    srand(time(NULL));
    dim3 grids(NUM_BLOCKS, Y_DIM);
    dim3 threads(MAX_THREADS_PER_BLOCK, 1);
    dim3 grids2(1, Y_DIM);
    dim3 threads2(MAX_THREADS_PER_BLOCK);
    float *d_vector, *h_vector;
    h_vector = (float *)malloc(Y_DIM * X_DIM * sizeof(float));
    memset(h_vector, 0, Y_DIM * X_DIM * sizeof(float));
    for (int i = 0; i < Y_DIM; i++)
        h_vector[i * X_DIM + i] = static_cast<float>(rand() % 101); // Genereer een willekeurig getal tussen 0 en 100
    cudaMalloc(&d_vector, Y_DIM * X_DIM * sizeof(float));
    cudaMemcpy(d_vector, h_vector, Y_DIM * X_DIM * sizeof(float), cudaMemcpyHostToDevice);

    int *max_index;
    int *max_val;
    int *d_max_index;
    int *d_max_val;

    max_index = (int *)malloc(Y_DIM * sizeof(int));
    max_val = (int *)malloc(Y_DIM * sizeof(int));
    cudaMalloc((void **)&d_max_index, Y_DIM * sizeof(int));
    cudaMalloc((void **)&d_max_val, Y_DIM * sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Eerste kernel
    cudaEventRecord(start);
    reduceBlockLevel<<<grids, threads>>>(d_vector, X_DIM, Y_DIM);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float et;
    cudaEventElapsedTime(&et, start, stop);
    printf("1ste kernel uitvoeringstijd: %fms\n", et);

    // Tweede kernel
    cudaEventRecord(start);
    reduceFinal<<<grids2, threads2>>>(d_max_index, d_max_val);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&et, start, stop);
    printf("2de kernel uitvoeringstijd: %fms\n", et);

    // Derde kernel
    int *d_global_max_index;
    int *d_globalMax;
    int *global_max_index = (int *)malloc(sizeof(int));
    int *globalMax = (int *)malloc(sizeof(int));
    cudaMalloc((void **)&d_global_max_index, sizeof(int));
    cudaMalloc((void **)&d_globalMax, sizeof(int));

    cudaEventRecord(start);
    reduceGlobal<<<1, MAX_THREADS_PER_BLOCK>>>(d_vector, X_DIM, Y_DIM, d_global_max_index, d_globalMax);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&et, start, stop);
    printf("3de kernel uitvoeringstijd: %fms\n", et);

    // Reduction Approach
    cudaEventRecord(start);
    reduceBlockLevel<<<grids, threads>>>(d_vector, X_DIM, Y_DIM);
    reduceFinal<<<grids2, threads2>>>(d_max_index, d_max_val);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&et, start, stop);
    printf("Reduction Approach uitvoeringstijd: %fms\n", et);

    cudaMemcpy(global_max_index, d_global_max_index, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(globalMax, d_globalMax, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Globaal max index: %d\n", *global_max_index);
    printf("Globaal max waarde: %d\n", *globalMax);

    // Naïeve CPU-benadering
    int naive_max_index;
    int naive_max_val;

    cudaEventRecord(start);
    naiveCPU(h_vector, X_DIM, Y_DIM, &naive_max_index, &naive_max_val);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&et, start, stop);
    printf("Naïeve CPU uitvoeringstijd: %fms\n", et);

    printf("Naïeve CPU Globaal max index: %d\n", naive_max_index);
    printf("Naïeve CPU Globaal max waarde: %d\n", naive_max_val);


    // AtomicMax kernel
    cudaEventRecord(start);
    atomicMaxKernel<<<1, MAX_THREADS_PER_BLOCK>>>(d_vector, X_DIM, Y_DIM, d_global_max_index, d_globalMax);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&et, start, stop);
    printf("AtomicMax kernel execution time: %fms\n", et);

    cudaMemcpy(global_max_index, d_global_max_index, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(globalMax, d_globalMax, sizeof(int), cudaMemcpyDeviceToHost);

    printf("AtomicMax Global max index: %d\n", *global_max_index);
    printf("AtomicMax Global max value: %d\n", *globalMax);


    free(max_index);
    free(max_val);
    free(global_max_index);
    free(globalMax);
    cudaFree(d_vector);
    cudaFree(d_max_index);
    cudaFree(d_max_val);
    cudaFree(d_global_max_index);
    cudaFree(d_globalMax);

    return 0;
}