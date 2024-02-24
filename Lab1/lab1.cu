#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <fstream>

#define THREADS_PER_BLOCK 256
#define MIN_ARRAY_LENGTH 1
#define MAX_ARRAY_LENGTH 1000000
#define NUM_MEASUREMENTS 10

void writeCSV(const char *filename, int arrayLength, float timeGlobal, float timeCPU)
{
    std::ofstream file(filename, std::ios_base::app | std::ios_base::out);
    file << arrayLength << "," << timeGlobal << "," << timeCPU << "\n";
    file.close();
}

__global__ void flipArrayGlobal(const int *input, int *output, int length)
{
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIdx < length / 2)
    {
        int oppositeIdx = length - globalIdx - 1;
        output[globalIdx] = input[oppositeIdx];
        output[oppositeIdx] = input[globalIdx];
    }
}

void flipArrayCPU(const int *input, int *output, int length)
{
    for (int i = 0; i < length / 2; ++i)
    {
        int temp = input[i];
        output[i] = input[length - i - 1];
        output[length - i - 1] = temp;
    }
}

int main()
{
    std::ofstream file("execution_times.csv");
    file << "ArrayLength,TimeGlobal,TimeCPU\n";
    file.close();

    for (int arrayLength = MIN_ARRAY_LENGTH; arrayLength <= MAX_ARRAY_LENGTH; arrayLength +=1000)
    {
        int *hostArray = (int *)malloc(arrayLength * sizeof(int));
        for (int i = 0; i < arrayLength; ++i)
        {
            hostArray[i] = i;
        }

        int *deviceArrayInput, *deviceArrayOutputGlobal;
        cudaMalloc((void **)&deviceArrayInput, arrayLength * sizeof(int));
        cudaMalloc((void **)&deviceArrayOutputGlobal, arrayLength * sizeof(int));

        cudaMemcpy(deviceArrayInput, hostArray, arrayLength * sizeof(int), cudaMemcpyHostToDevice);

        float totalMillisecondsGlobal = 0;
        float totalMillisecondsCPU = 0;

        for (int measurement = 0; measurement < NUM_MEASUREMENTS; ++measurement)
        {
 
            int numBlocks = (arrayLength + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

            cudaEvent_t startGlobal, stopGlobal;
            cudaEventCreate(&startGlobal);
            cudaEventCreate(&stopGlobal);

            cudaEventRecord(startGlobal);
            flipArrayGlobal<<<numBlocks, THREADS_PER_BLOCK>>>(deviceArrayInput, deviceArrayOutputGlobal, arrayLength);
            cudaEventRecord(stopGlobal);

            cudaEventSynchronize(stopGlobal);
            float millisecondsGlobal = 0;
            cudaEventElapsedTime(&millisecondsGlobal, startGlobal, stopGlobal);
            totalMillisecondsGlobal += millisecondsGlobal;

            cudaMemcpy(hostArray, deviceArrayOutputGlobal, arrayLength * sizeof(int), cudaMemcpyDeviceToHost);

            int *hostArrayOutputCPU = (int *)malloc(arrayLength * sizeof(int));

            cudaEvent_t startCPU, stopCPU;
            cudaEventCreate(&startCPU);
            cudaEventCreate(&stopCPU);

            cudaEventRecord(startCPU);
            flipArrayCPU(hostArray, hostArrayOutputCPU, arrayLength);
            cudaEventRecord(stopCPU);

            cudaEventSynchronize(stopCPU);
            float millisecondsCPU = 0;
            cudaEventElapsedTime(&millisecondsCPU, startCPU, stopCPU);
            totalMillisecondsCPU += millisecondsCPU;

            free(hostArrayOutputCPU);
        }

        float averageMillisecondsGlobal = totalMillisecondsGlobal / NUM_MEASUREMENTS;
        float averageMillisecondsCPU = totalMillisecondsCPU / NUM_MEASUREMENTS;

        writeCSV("execution_times.csv", arrayLength, averageMillisecondsGlobal, averageMillisecondsCPU);

        free(hostArray);
        cudaFree(deviceArrayInput);
        cudaFree(deviceArrayOutputGlobal);
    }

    return 0;
}
