#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>


__global__ void grayscale_coalesced(unsigned char* image, int width, int height) {
    int idx_start = (blockIdx.x * blockDim.x + threadIdx.x) * 3;
    int threadCount = gridDim.x * blockDim.x;

    for (int idx = idx_start; idx < width * height * 3; idx += threadCount * 3) {
        unsigned char value = (image[idx] + image[idx + 1] + image[idx + 2]) / 3;
        image[idx] = value;
        image[idx + 1] = value;
        image[idx + 2] = value;
    }
}

__global__ void grayscale_notCoalesced(unsigned char* image, int width, int height) {
    int idx_start = (blockIdx.x * blockDim.x + threadIdx.x) * 3;
    int threadCount = gridDim.x * blockDim.x;

    for (int idx = idx_start; idx < width * height; idx += threadCount) {
        int pixelIdx = idx;
        unsigned char value = (image[pixelIdx] + image[pixelIdx + width * height] + image[pixelIdx + 2 * width * height]) / 3;
        image[pixelIdx] = value;
        image[pixelIdx + width * height] = value;
        image[pixelIdx + 2 * width * height] = value;
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
    // Specify input and output file names
    std::string inputFilename = "C:/Users/vanni/source/repos/CUDA_Lab3/image-import/input_image.ppm";
    std::string outputFilename1 = "output_image1.ppm";
    std::string outputFilename2 = "output_image2.ppm";
    std::string csvFilename = "C:/Users/vanni/Desktop/School/times.csv";

    // Read input image
    int width, height;
    std::vector<unsigned char> h_image;
    readPPM(inputFilename, h_image, width, height);

    // Allocate memory for the images on the device
    unsigned char* d_image1;
    unsigned char* d_image2;
    cudaMalloc((void**)&d_image1, h_image.size());
    cudaMalloc((void**)&d_image2, h_image.size());
    cudaMemcpy(d_image1, h_image.data(), h_image.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_image2, h_image.data(), h_image.size(), cudaMemcpyHostToDevice);

    // Specify the block and grid dimensions
    int numBlocks = 4;

    // Open CSV file for writing
    std::ofstream csvFile(csvFilename);
    if (!csvFile.is_open()) {
        std::cerr << "Error opening CSV file for writing." << std::endl;
        return 1;
    }

    // Write CSV header
    csvFile << "Block Size,Time (Coalesced),Time (Not Coalesced)\n";

    // Loop over different block sizes
    for (int blockSize = 1; blockSize < 1024; blockSize++) {
        float milliseconds1_Total = 0;
        float milliseconds2_Total = 0;
        int amountOfTries = 10;

        std::cout << blockSize << std::endl;
        for (int i = 0; i < amountOfTries; i++) {
            // Timing variables
            cudaEvent_t start1, stop1, start2, stop2;
            cudaEventCreate(&start1);
            cudaEventCreate(&stop1);
            cudaEventCreate(&start2);
            cudaEventCreate(&stop2);

            // Record start time for kernel 1
            cudaEventRecord(start1);

            // Launch the kernel for image processing operation 1 (e.g., grayscale)
            grayscale_coalesced << <numBlocks, blockSize >> > (d_image1, width, height);

            // Record stop time for kernel 1
            cudaEventRecord(stop1);
            cudaEventSynchronize(stop1);

            // Calculate and print the elapsed time for kernel 1
            float milliseconds1 = 0;
            cudaEventElapsedTime(&milliseconds1, start1, stop1);
            milliseconds1_Total += milliseconds1;

            // Record start time for kernel 2
            cudaEventRecord(start2);

            // Launch the kernel for image processing operation 2 (e.g., invertImageKernel)
            grayscale_notCoalesced << <numBlocks, blockSize >> > (d_image2, width, height);

            // Record stop time for kernel 2
            cudaEventRecord(stop2);
            cudaEventSynchronize(stop2);

            // Calculate and print the elapsed time for kernel 2
            float milliseconds2 = 0;
            cudaEventElapsedTime(&milliseconds2, start2, stop2);
            milliseconds2_Total += milliseconds2;
        }
        csvFile << blockSize << "," << milliseconds1_Total/amountOfTries << "," << milliseconds2_Total/amountOfTries << "\n";
    }

    // Close CSV file
    csvFile.close();

    // Copy the processed images back to the host
    std::vector<unsigned char> h_image1(h_image.size());
    std::vector<unsigned char> h_image2(h_image.size());
    cudaMemcpy(h_image1.data(), d_image1, h_image.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_image2.data(), d_image2, h_image.size(), cudaMemcpyDeviceToHost);

    // Write the output images
    //writePPM(outputFilename1, h_image1.data(), width, height);
    //writePPM(outputFilename2, h_image2.data(), width, height);

    // Free allocated memory
    cudaFree(d_image1);
    cudaFree(d_image2);

    return 0;
}