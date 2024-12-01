#include "../common/cuda_utils.cuh"
#include <fstream>
#include <vector>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>

// Structure to hold our pairs of numbers on both CPU and GPU
struct NumberPair {
    int first;
    int second;
};

std::vector<NumberPair> readNumberPairs(const char* filename) {
    std::vector<NumberPair> pairs;
    std::ifstream file(filename);
    std::string line;
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        NumberPair pair;
        if (iss >> pair.first >> pair.second) {
            pairs.push_back(pair);
        }
    }
    
    return pairs;
}

__global__ void processPairs(const NumberPair* pairs, int* firsts, int* seconds, int count) {
    // Calculate which pair this thread should process
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < count) {
        // Here you can perform your specific operation on the pair
        // For example, let's calculate the sum of each pair
        firsts[idx] = pairs[idx].first;
        seconds[idx] = pairs[idx].second;

    }
}

__global__ void findDiscrepency(int* firsts, int* seconds, int* results, int count) {
    // Calculate which pair this thread should process
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < count) {
        // Here you can perform your specific operation on the pair
        // For example, let's calculate the sum of each pair
        results[idx] = abs(firsts[idx] - seconds[idx]);
        
    }
}

__global__ void findAndAddDuplicate(int* firsts, int* seconds, int* results, int count) {
    // Calculate which pair this thread should process
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < count) {
        // Here you can perform your specific operation on the pair
        // For example, let's calculate the sum of each pair
        results[idx] = 0;
        for (int i=0; i<count;i++){
            if (firsts[idx] == seconds[i]){
                results[idx] += seconds[i];
            }
        }
    }
}

int sumWithThrust(int* d_data, int count){
    thrust::device_ptr<int> thrust_data(d_data);
    return thrust::reduce(thrust_data, thrust_data+count);
}

void sortWithThrust(int* d_data, int count) {
    // Create device pointers and sort from host code
    thrust::device_ptr<int> thrust_data(d_data);
    thrust::sort(thrust_data, thrust_data + count);
}

void puzzle_1(int pairCount, std::vector<NumberPair> &pairs);
void puzzle_2(int pairCount, std::vector<NumberPair> &pairs);

int main()
{

    char cwd[1024];
    chdir("../day01/");
    getcwd(cwd, sizeof(cwd));
    printf("Current working dir: %s\n", cwd);
    std::vector<NumberPair> pairs = readNumberPairs("input.txt");
    for(int i = 0; i<pairs.size(); i++){
        std::cout<<pairs[i].first<<" "<<pairs[i].second<< std::endl;
    }
    int pairCount = pairs.size();

    puzzle_1(pairCount, pairs);

    puzzle_2(pairCount, pairs);

    return 0;
}

void puzzle_1(int pairCount, std::vector<NumberPair> &pairs)
{
    NumberPair *d_pairs;
    int *d_results;
    int *d_firsts;
    int *d_seconds;
    CUDA_CHECK(cudaMalloc(&d_pairs, pairCount * sizeof(NumberPair)));
    CUDA_CHECK(cudaMalloc(&d_results, pairCount * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_firsts, pairCount * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seconds, pairCount * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_pairs, pairs.data(),
                          pairCount * sizeof(NumberPair),
                          cudaMemcpyHostToDevice));
    int blockSize = 256;
    int numBlocks = (pairCount + blockSize - 1) / blockSize;
    processPairs<<<numBlocks, blockSize>>>(d_pairs, d_firsts, d_seconds, pairCount);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Process 1 complete" << std::endl;
    sortWithThrust(d_firsts, pairCount);
    sortWithThrust(d_seconds, pairCount);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Process 2 complete" << std::endl;

    findDiscrepency<<<numBlocks, blockSize>>>(d_firsts, d_seconds, d_results, pairCount);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Process 3 complete" << std::endl;
    int total_sum = sumWithThrust(d_results, pairCount);

    std::cout << "Process 4 complete" << std::endl;
    std::cout << total_sum << std::endl;

    CUDA_CHECK(cudaFree(d_pairs));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_firsts));
    CUDA_CHECK(cudaFree(d_seconds));
}

void puzzle_2(int pairCount, std::vector<NumberPair> &pairs)
{
    NumberPair *d_pairs;
    int *d_results;
    int *d_firsts;
    int *d_seconds;
    CUDA_CHECK(cudaMalloc(&d_pairs, pairCount * sizeof(NumberPair)));
    CUDA_CHECK(cudaMalloc(&d_results, pairCount * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_firsts, pairCount * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seconds, pairCount * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_pairs, pairs.data(),
                          pairCount * sizeof(NumberPair),
                          cudaMemcpyHostToDevice));
    int blockSize = 256;
    int numBlocks = (pairCount + blockSize - 1) / blockSize;
    processPairs<<<numBlocks, blockSize>>>(d_pairs, d_firsts, d_seconds, pairCount);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Process 1 complete" << std::endl;


    findAndAddDuplicate<<<numBlocks, blockSize>>>(d_firsts, d_seconds, d_results, pairCount);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Process 3 complete" << std::endl;
    int total_sum = sumWithThrust(d_results, pairCount);

    std::cout << "Process 4 complete" << std::endl;
    std::cout << total_sum << std::endl;

    CUDA_CHECK(cudaFree(d_pairs));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_firsts));
    CUDA_CHECK(cudaFree(d_seconds));
}