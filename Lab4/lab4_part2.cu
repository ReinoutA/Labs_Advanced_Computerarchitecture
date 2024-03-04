#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

const int MAX_N = 78; // Maximum array size
const int TILE_SIZE = 78;
const int NUM_MEASUREMENT = 1000;
const int THREADS_PER_BLOCK = 16;

// Matrix multiplication kernel using global memory only
__global__ void matrixMulGlobal(float* A, float* B, float* C, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int k = 0; k < N; ++k) {
        sum += A[i * N + k] * B[k * N + j];
    }

    C[i * N + j] = sum;
}

// Matrix multiplication kernel using global and shared memory
__global__ void matrixMulGlobalShared(float* A, float* B, float* C, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for (int tileIndex = 0; tileIndex < N / TILE_SIZE; ++tileIndex) {
        tileA[threadIdx.y][threadIdx.x] = A[i * N + tileIndex * TILE_SIZE + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = B[j + N * (tileIndex * TILE_SIZE + threadIdx.y)];

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    C[i * N + j] = sum;
}

// Declare constant memory
__constant__ float c_A[MAX_N * MAX_N];
__constant__ float c_B[MAX_N * MAX_N];

// Matrix multiplication kernel using global and constant memory
__global__ void matrixMulGlobalConstant(float* C, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int k = 0; k < N; ++k) {
        sum += c_A[i * N + k] * c_B[k * N + j];
    }

    C[i * N + j] = sum;
}

void testKernels() {

    double n = 72382.413651;
    int len;

    len = sizeof(n);
    printf("%d\n", len);


    const int N = 2;

    float h_A[N * N] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_B[N * N] = {5.0f, 6.0f, 7.0f, 8.0f};
    float h_C_global[N * N] = {0.0f};
    float h_C_shared[N * N] = {0.0f};
    float h_C_constant[N * N] = {0.0f};

    float *d_A, *d_B, *d_C_global, *d_C_shared, *d_C_constant;
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C_global, N * N * sizeof(float));
    cudaMalloc((void**)&d_C_shared, N * N * sizeof(float));
    cudaMalloc((void**)&d_C_constant, N * N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_A, h_A, N * N * sizeof(float));
    cudaMemcpyToSymbol(c_B, h_B, N * N * sizeof(float));

    dim3 blockDim(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);


    matrixMulGlobal<<<gridDim, blockDim>>>(d_A, d_B, d_C_global, N);
    matrixMulGlobalShared<<<gridDim, blockDim>>>(d_A, d_B, d_C_shared, N);
    matrixMulGlobalConstant<<<gridDim, blockDim>>>(d_C_constant, N);

    // Copy results from device to host
    cudaMemcpy(h_C_global, d_C_global, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_C_shared, d_C_global, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(h_C_shared, d_C_shared, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_constant, d_C_constant, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Global Memory Result:" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << h_C_global[i * N + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Shared Memory Result:" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << h_C_shared[i * N + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Constant Memory Result:" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << h_C_constant[i * N + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_global);
    cudaFree(d_C_shared);
    cudaFree(d_C_constant);
}


int main() {
    // Test the kernels with a small input size
    testKernels();

    // Open CSV file for writing
    std::ofstream csvFile("execution_times.csv");
    csvFile << "ArraySize,GlobalMemoryTime(ms),SharedMemoryTime(ms),ConstantMemoryTime(ms)\n";

    for (int currentN = 1; currentN <= MAX_N; currentN++) {
        // Initialize matrices on the host
        float* h_A = new float[currentN * currentN];
        float* h_B = new float[currentN * currentN];
        float* h_C_global = new float[currentN * currentN];
        float* h_C_shared = new float[currentN * currentN];
        float* h_C_constant = new float[currentN * currentN];

        // Initialize matrices with random values
        for (int i = 0; i < currentN * currentN; ++i) {
            h_A[i] = static_cast<float>(rand()) / RAND_MAX;
            h_B[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        // Allocate memory on the device
        float *d_A, *d_B, *d_C_global, *d_C_shared, *d_C_constant;
        cudaMalloc((void**)&d_A, currentN * currentN * sizeof(float));
        cudaMalloc((void**)&d_B, currentN * currentN * sizeof(float));
        cudaMalloc((void**)&d_C_global, currentN * currentN * sizeof(float));
        cudaMalloc((void**)&d_C_shared, currentN * currentN * sizeof(float));
        cudaMalloc((void**)&d_C_constant, currentN * currentN * sizeof(float));

        // Copy matrices from host to device
        cudaMemcpy(d_A, h_A, currentN * currentN * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, currentN * currentN * sizeof(float), cudaMemcpyHostToDevice);

        // Set up grid and block dimensions
        dim3 blockDim(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
        dim3 gridDim((currentN + blockDim.x - 1) / blockDim.x, (currentN + blockDim.y - 1) / blockDim.y);

        // Measure time for global memory only
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float globalTotalTime = 0.0f;
        for (int i = 0; i < NUM_MEASUREMENT; ++i) {
            cudaEventRecord(start);
            matrixMulGlobal<<<gridDim, blockDim>>>(d_A, d_B, d_C_global, currentN);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float globalTime;
            cudaEventElapsedTime(&globalTime, start, stop);

            globalTotalTime += globalTime;
        }

        float globalAverageTime = globalTotalTime / NUM_MEASUREMENT;

        // Measure time for global and shared memory
        float sharedTotalTime = 0.0f;
        for (int i = 0; i < NUM_MEASUREMENT; ++i) {
            cudaEventRecord(start);
            matrixMulGlobalShared<<<gridDim, blockDim>>>(d_A, d_B, d_C_shared, currentN);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float sharedTime;
            cudaEventElapsedTime(&sharedTime, start, stop);

            sharedTotalTime += sharedTime;
        }

        float sharedAverageTime = sharedTotalTime / NUM_MEASUREMENT;

        // Measure time for global and constant memory
        float constantTotalTime = 0.0f;
        for (int i = 0; i < NUM_MEASUREMENT; ++i) {
            cudaEventRecord(start);
            matrixMulGlobalConstant<<<gridDim, blockDim>>>(d_C_constant, currentN);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float constantTime;
            cudaEventElapsedTime(&constantTime, start, stop);

            constantTotalTime += constantTime;
        }

        float constantAverageTime = constantTotalTime / NUM_MEASUREMENT;

        // Write results to CSV file
        csvFile << currentN << "," << globalAverageTime << "," << sharedAverageTime << "," << constantAverageTime << "\n";

        // Cleanup for the current iteration
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C_global);
        cudaFree(d_C_shared);
        cudaFree(d_C_constant);

        delete[] h_A;
        delete[] h_B;
        delete[] h_C_global;
        delete[] h_C_shared;
        delete[] h_C_constant;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Close CSV file
    csvFile.close();
   
    return 0;
}



