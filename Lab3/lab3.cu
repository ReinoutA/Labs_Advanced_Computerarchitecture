#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

// Kernel to invert an image (RGB components)
__global__ void invertImageKernel(unsigned char* image, int width, int height) {
    int threadCount = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < width * height * 3; idx += threadCount) {
        image[idx] = 255 - image[idx];
    }
}

// New kernel function with additional conditions
__global__ void invertImageKernel_R(unsigned char* image, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int threadCount = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < width * height * 3; idx += threadCount) {
        if ((idx % 3 == 0) && (image[idx] > 100) && (image[idx] < 200)) {
            image[idx] = (image[idx] % 25) * 10;
        }
        else {
            image[idx] = 255 - image[idx];
        }
    }
}

// New kernel function to process data in the RR...RGGG...GBB...B format
__global__ void invertImageKernel_NewFormat(unsigned char* image, int width, int height) {
    int threadCount = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < width * height * 3; idx += threadCount) {
        // Determine the color channel (R, G, or B)
        int channel = idx % 3;

        // Determine the position within the new format
        int newPos = idx / 3;

        // Calculate the new index based on the position within the new format and the color channel
        int newIdx = (newPos * 3) + channel;

        if ((channel == 0) && (image[idx] > 100) && (image[idx] < 200)) {
            image[idx] = (image[idx] % 25) * 10;
        }
        else {
            image[idx] = 255 - image[idx];
        }
    }
}

void readPPM(const std::string& filename, std::vector<unsigned char>& image, int& width, int& height) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string format;
    file >> format >> width >> height;
    int maxVal;
    file >> maxVal;

    image.resize(width * height * 3);
    file.read(reinterpret_cast<char*>(image.data()), image.size());
    file.close();
}

void writePPM(const std::string& filename, const std::vector<unsigned char>& image, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    file << "P6\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(image.data()), image.size());
    file.close();
}

int main() {
    int numBlocks = 16;
    std::string inputFilename = "input_image.ppm";
    std::string outputFilename = "output_image.ppm";
    std::string csvFilename = "timing_results.csv";
    int width, height;
    std::vector<unsigned char> h_image;
    readPPM(inputFilename, h_image, width, height);

    unsigned char* d_image;
    cudaMalloc((void**)&d_image, h_image.size());
    cudaMemcpy(d_image, h_image.data(), h_image.size(), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    std::vector<float> timingResultsKernel;

    // Loop to progressively use more threads
    for (int threadsPerBlock = 1; threadsPerBlock <= 1024; threadsPerBlock++) {
        float totalMilliseconds = 0;

        for (int i = 0; i < 10; i++) {
            //int numBlocks = (h_image.size() + threadsPerBlock - 1) / threadsPerBlock;

            cudaEventRecord(start);

            invertImageKernel<<<numBlocks, threadsPerBlock>>>(d_image, width, height);
            cudaDeviceSynchronize(); 

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            totalMilliseconds += milliseconds;
        }

        float averageMilliseconds = totalMilliseconds / 10;
        timingResultsKernel.push_back(averageMilliseconds);

        std::cout << "Threads Per Block (Kernel): " << threadsPerBlock << ", Average Time taken: " << averageMilliseconds << " ms" << std::endl;
    }

    std::vector<float> timingResultsKernel_R;

    // Loop to progressively use more threads
    for (int threadsPerBlock = 1; threadsPerBlock <= 1024; threadsPerBlock++) {
        float totalMilliseconds = 0;

        for (int i = 0; i < 10; i++) {            
            //int numBlocks = (h_image.size() + threadsPerBlock - 1) / threadsPerBlock;

            cudaEventRecord(start);
            invertImageKernel_R<<<numBlocks, threadsPerBlock>>>(d_image, width, height);
            cudaDeviceSynchronize();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            totalMilliseconds += milliseconds;
        }

        float averageMilliseconds = totalMilliseconds / 10;
        timingResultsKernel_R.push_back(averageMilliseconds);

        std::cout << "Threads Per Block (Kernel_R): " << threadsPerBlock << ", Average Time taken: " << averageMilliseconds << " ms" << std::endl;
    }

    std::vector<float> timingResultsKernel_NewFormat;

    // Loop to progressively use more threads
    for (int threadsPerBlock = 1; threadsPerBlock <= 1024; threadsPerBlock++) {
        float totalMilliseconds = 0;
        for (int i = 0; i < 10; i++) {
            //int numBlocks = (h_image.size() + threadsPerBlock - 1) / threadsPerBlock;

            cudaEventRecord(start);

            invertImageKernel_NewFormat<<<numBlocks, threadsPerBlock>>>(d_image, width, height);
            cudaDeviceSynchronize();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            totalMilliseconds += milliseconds;
        }

        float averageMilliseconds = totalMilliseconds / 10;
        timingResultsKernel_NewFormat.push_back(averageMilliseconds);

        std::cout << "Threads Per Block (Kernel_NewFormat): " << threadsPerBlock << ", Average Time taken: " << averageMilliseconds << " ms" << std::endl;
    }

    std::ofstream csvFile(csvFilename);
    if (csvFile.is_open()) {
        csvFile << "ThreadsPerBlock,AverageTime_Kernel(ms),AverageTime_Kernel_R(ms),AverageTime_Kernel_NewFormat(ms)\n";
        for (int i = 0; i < timingResultsKernel.size(); i++) {
            csvFile << i + 1 << "," << timingResultsKernel[i] << "," << timingResultsKernel_R[i] << "," << timingResultsKernel_NewFormat[i] << "\n";
        }
        csvFile.close();
        std::cout << "Timing results written to: " << csvFilename << std::endl;
    } else {
        std::cerr << "Error opening CSV file for writing." << std::endl;
    }

    cudaMemcpy(h_image.data(), d_image, h_image.size(), cudaMemcpyDeviceToHost);
    writePPM(outputFilename, h_image, width, height);
    cudaFree(d_image);

    return 0;
}
