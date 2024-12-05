#include "../common/cuda_utils.cuh"
#include <fstream>
#include <vector>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <numeric>
#include <string>
#include <regex>

std::string readString(const char* filename) {
    std::vector<std::string> stringVector;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string lineString = iss.str();
        stringVector.push_back(lineString);
    }
    std::string stringData = std::accumulate(stringVector.begin(), stringVector.end(), std::string{});

    
    return stringData;
}

__global__ void processStringMul(const char* data, int* results, int count) {
    // Calculate which pair this thread should process
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const char* mulSearchString = "mul(";
    const int mulSearchStringSize = 4;
    const char* numberSearchString = "0123456789";
    const int numberSearchStringSize = 10;
    
    if (idx < count-mulSearchStringSize) {
        results[idx] = 1;
        for (int i = 0; i<mulSearchStringSize; i++){
            if (data[idx+i]!=mulSearchString[i]){
                results[idx]=0;
            } 
        }
        if (results[idx]==1){
            bool numberMatch = true;
            int numberLength = 0;
            int oldNumberLength = 0;
            int number_1 = 0;
            while (numberMatch){
                oldNumberLength = numberLength;
                char numberCandidate = data[idx+mulSearchStringSize+numberLength];
                for(int i=0; i<numberSearchStringSize;i++){    
                    if (numberCandidate==numberSearchString[i]){
                        number_1 = number_1*10+(numberCandidate-'0');
                        numberLength++;
                        break;
                    }
                }
                if (numberLength==oldNumberLength){
                    numberMatch = false;
                }
            }
            if (number_1 == 0){
                results[idx] = 0;
            } else {
                int curr_idx = mulSearchStringSize+numberLength;
                if (data[idx+curr_idx] == ','){
                    curr_idx += 1;
                    bool numberMatch = true;
                    int numberLength = 0;
                    int oldNumberLength = 0;
                    int number_2 = 0;
                    while (numberMatch){
                        oldNumberLength = numberLength;
                        char numberCandidate = data[idx+curr_idx+numberLength];
                        for(int i=0; i<numberSearchStringSize;i++){    
                            if (numberCandidate==numberSearchString[i]){
                                number_2 = number_2*10+(numberCandidate-'0');
                                numberLength++;
                                break;
                            }
                        }
                        if (numberLength==oldNumberLength){
                            numberMatch = false;
                        }
                    }
                    if (number_2 == 0 || data[idx+curr_idx+numberLength] != ')'){
                        results[idx] = 0;
                    } else{
                        results[idx] = number_1 * number_2;
                    }

                } else {
                    results[idx] = 0;
                }
            }
            
        }

    }
}

__global__ void processStringMulConditional(const char* data, const bool* allowed_sites, int* results, int count) {
    // Calculate which pair this thread should process
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const char* mulSearchString = "mul(";
    const int mulSearchStringSize = 4;
    const char* numberSearchString = "0123456789";
    const int numberSearchStringSize = 10;
    
    if (idx < count-mulSearchStringSize) {
        if(allowed_sites[idx]){
            results[idx] = 1;
            for (int i = 0; i<mulSearchStringSize; i++){
                if (data[idx+i]!=mulSearchString[i]){
                    results[idx]=0;
                } 
            }
            if (results[idx]==1){
                bool numberMatch = true;
                int numberLength = 0;
                int oldNumberLength = 0;
                int number_1 = 0;
                while (numberMatch){
                    oldNumberLength = numberLength;
                    char numberCandidate = data[idx+mulSearchStringSize+numberLength];
                    for(int i=0; i<numberSearchStringSize;i++){    
                        if (numberCandidate==numberSearchString[i]){
                            number_1 = number_1*10+(numberCandidate-'0');
                            numberLength++;
                            break;
                        }
                    }
                    if (numberLength==oldNumberLength){
                        numberMatch = false;
                    }
                }
                if (number_1 == 0){
                    results[idx] = 0;
                } else {
                    int curr_idx = mulSearchStringSize+numberLength;
                    if (data[idx+curr_idx] == ','){
                        curr_idx += 1;
                        bool numberMatch = true;
                        int numberLength = 0;
                        int oldNumberLength = 0;
                        int number_2 = 0;
                        while (numberMatch){
                            oldNumberLength = numberLength;
                            char numberCandidate = data[idx+curr_idx+numberLength];
                            for(int i=0; i<numberSearchStringSize;i++){    
                                if (numberCandidate==numberSearchString[i]){
                                    number_2 = number_2*10+(numberCandidate-'0');
                                    numberLength++;
                                    break;
                                }
                            }
                            if (numberLength==oldNumberLength){
                                numberMatch = false;
                            }
                        }
                        if (number_2 == 0 || data[idx+curr_idx+numberLength] != ')'){
                            results[idx] = 0;
                        } else{
                            results[idx] = number_1 * number_2;
                        }

                    } else {
                        results[idx] = 0;
                    }
                }
                
            }
        } else {
            results[idx] = 0;
        }
    }
}


// __global__ void findDiscrepency(int* firsts, int* seconds, int* results, int count) {
//     // Calculate which pair this thread should process
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
//     if (idx < count) {
//         // Here you can perform your specific operation on the pair
//         // For example, let's calculate the sum of each pair
//         results[idx] = abs(firsts[idx] - seconds[idx]);
        
//     }
// }

// __global__ void findAndAddDuplicate(int* firsts, int* seconds, int* results, int count) {
//     // Calculate which pair this thread should process
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
//     if (idx < count) {
//         // Here you can perform your specific operation on the pair
//         // For example, let's calculate the sum of each pair
//         results[idx] = 0;
//         for (int i=0; i<count;i++){
//             if (firsts[idx] == seconds[i]){
//                 results[idx] += seconds[i];
//             }
//         }
//     }
// }

int sumWithThrust(int* d_data, int count){
    thrust::device_ptr<int> thrust_data(d_data);
    return thrust::reduce(thrust_data, thrust_data+count);
}

// void sortWithThrust(int* d_data, int count) {
//     // Create device pointers and sort from host code
//     thrust::device_ptr<int> thrust_data(d_data);
//     thrust::sort(thrust_data, thrust_data + count);
// }

void puzzle_1(std::string &stringData);
void puzzle_2(std::string &stringData);

// void puzzle_2(int pairCount, std::vector<NumberPair> &pairs);

int main()
{

    char cwd[1024];
    chdir("../day03/");
    getcwd(cwd, sizeof(cwd));
    printf("Current working dir: %s\n", cwd);
    std::string stringData = readString("input.txt");
    std::cout<<stringData<<std::endl;
    // for (int i: dataVector){
    //     std::cout << i << ' ';        
    // }
    // std::cout<<std::endl;
    puzzle_1(stringData);
    puzzle_2(stringData);

    return 0;
}

void puzzle_1(std::string &stringData)
{
    // std::vector<int> results(rowLengthVector.size());
    char *d_data;
    int *d_string_candidates;
    CUDA_CHECK(cudaMalloc(&d_data, stringData.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_string_candidates, stringData.size() * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_data, stringData.data(),
                          stringData.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (stringData.size() + blockSize - 1) / blockSize;
    processStringMul<<<numBlocks, blockSize>>>(d_data, d_string_candidates, stringData.size());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Process 1 complete" << std::endl;

    int total_sum = sumWithThrust(d_string_candidates, stringData.size());

    std::cout << "Process 2 complete" << std::endl;
    std::cout << total_sum << std::endl;
    CUDA_CHECK(cudaDeviceSynchronize());

    // CUDA_CHECK(cudaMemcpy(&results[0], d_results, rowLengthVector.size()*sizeof(int), cudaMemcpyDeviceToHost));
    // for (int i: results){
    //     std::cout << i <<" ";
    // }
    // std::cout<<std::endl;
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_string_candidates));
}

void puzzle_2(std::string &stringData)
{
    bool allowed_sites[stringData.size()];
    bool do_found = true;
    std::fill(allowed_sites, allowed_sites+ stringData.size(), true);
    for (int i=0; i<stringData.size()-7; i++){
        if (do_found and strcmp(stringData.substr(i, 7).data(), "don't()")==0){
            std::cout<<"Found don't at "<<i<<std::endl;
            do_found = false;
            std::fill(allowed_sites+i, allowed_sites+stringData.size(), false);
        } else if (!do_found and strcmp(stringData.substr(i,4).data(), "do()")==0){
            std::cout<<"Found do at "<<i<<std::endl;
            do_found = true;
            std::fill(allowed_sites+i, allowed_sites+stringData.size(), true);
        }
    }

    // std::vector<int> results(rowLengthVector.size());
    char *d_data;
    bool *d_allowed_sites;
    int *d_string_candidates;
    CUDA_CHECK(cudaMalloc(&d_data, stringData.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_string_candidates, stringData.size() * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_allowed_sites, stringData.size() * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_data, stringData.data(),
                          stringData.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_allowed_sites, allowed_sites,
                          stringData.size() * sizeof(bool),
                          cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (stringData.size() + blockSize - 1) / blockSize;
    processStringMulConditional<<<numBlocks, blockSize>>>(d_data, d_allowed_sites, d_string_candidates, stringData.size());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Process 1 complete" << std::endl;

    int total_sum = sumWithThrust(d_string_candidates, stringData.size());

    std::cout << "Process 2 complete" << std::endl;
    std::cout << total_sum << std::endl;
    CUDA_CHECK(cudaDeviceSynchronize());

    // CUDA_CHECK(cudaMemcpy(&results[0], d_results, rowLengthVector.size()*sizeof(int), cudaMemcpyDeviceToHost));
    // for (int i: results){
    //     std::cout << i <<" ";
    // }
    // std::cout<<std::endl;
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_string_candidates));
}