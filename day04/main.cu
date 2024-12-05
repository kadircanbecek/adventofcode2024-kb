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

struct StringMap{
    std::string stringData;
    int dim1;
    int dim2;
};

StringMap readString(const char* filename) {
    StringMap stringMap;
    std::vector<std::string> stringVector;
    std::ifstream file(filename);
    std::string line;
    int dim1=0, dim2 = 0;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string lineString = iss.str();
        if (dim2 == 0){
            dim2 = lineString.size();
            for (int i = 0 ; i < 3; i++){
                stringVector.push_back(std::string(dim2+6, '.'));
                dim1++;
            }
        }
        stringVector.push_back("..."+lineString+"...");
        dim1++;
    }
    for (int i = 0 ; i < 3; i++){
        stringVector.push_back(std::string(dim2+6, '.'));
        dim1++;
    }
    std::string stringData = std::accumulate(stringVector.begin(), stringVector.end(), std::string{});
    stringMap.stringData = stringData;
    stringMap.dim1 = dim1;
    stringMap.dim2 = dim2+6;
    
    return stringMap;
}

__global__ void processStringXMAS(const char* data, int* results, int dim1, int dim2) {
    // Calculate which pair this thread should process
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < (dim1-6)*(dim2-6)) {
        results[idx]=0;
        int idx_dim1, idx_dim2, all_idx;
        idx_dim1 = idx/(dim2-6)+3;
        idx_dim2 = idx%(dim2-6)+3;
        all_idx = idx_dim1*dim2+idx_dim2;
        
        if (data[all_idx] == 'X'){
            // results[idx]=1;
            // return;
            int ind_in1_1, ind_in1_2, all_ind1, all_ind2, all_ind3;
            for (int i=0; i<9; i++){
                ind_in1_1 = i/3-1;
                ind_in1_2 = i%3-1;
                all_ind1 = (idx_dim1+ind_in1_1)*dim2+idx_dim2+ind_in1_2;
                if (data[all_ind1]=='M'){
                    all_ind2 = (idx_dim1+ind_in1_1*2)*dim2+idx_dim2+ind_in1_2*2;
                        if (data[all_ind2]=='A'){
                            all_ind3 = (idx_dim1+ind_in1_1*3)*dim2+idx_dim2+ind_in1_2*3;
                            if (data[all_ind3]=='S'){
                                results[idx]++;
                            }
                        }
                }
            }
        }
    }
}

__global__ void processStringX_MAS(const char* data, int* results, int dim1, int dim2) {
    // Calculate which pair this thread should process
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < (dim1-6)*(dim2-6)) {
        results[idx]=0;
        int idx_dim1, idx_dim2, all_idx;
        idx_dim1 = idx/(dim2-6)+3;
        idx_dim2 = idx%(dim2-6)+3;
        all_idx = idx_dim1*dim2+idx_dim2;
        
        if (data[all_idx] == 'A'){
            int ind_in1_1, ind_in1_2, all_ind1, all_ind2;
            for (int i=0; i<9; i++){
                ind_in1_1 = i/3-1;
                ind_in1_2 = i%3-1;
                if(ind_in1_1!=0 and ind_in1_2!=0){
                    all_ind1 = (idx_dim1+ind_in1_1)*dim2+idx_dim2+ind_in1_2;
                    if (data[all_ind1]=='M'){
                        all_ind2 = (idx_dim1-ind_in1_1)*dim2+idx_dim2-ind_in1_2;
                        if(data[all_ind2]=='S'){
                            results[idx]++;
                        }
                    }
                }
            }
        }
        if (results[idx]==2){
            results[idx] = 1;
        }else{
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

void puzzle_1(StringMap &stringMap);
void puzzle_2(StringMap &stringMap);

// void puzzle_2(int pairCount, std::vector<NumberPair> &pairs);

int main()
{

    char cwd[1024];
    chdir("../day04/");
    getcwd(cwd, sizeof(cwd));
    printf("Current working dir: %s\n", cwd);
    StringMap stringMap = readString("input.txt");
    std::cout<<stringMap.stringData<<std::endl;
    std::cout<<stringMap.dim1<<std::endl;
    std::cout<<stringMap.dim2<<std::endl;
    // for (int i: dataVector){
    //     std::cout << i << ' ';        
    // }
    // std::cout<<std::endl;
    puzzle_1(stringMap);
    puzzle_2(stringMap);

    return 0;
}

void puzzle_1(StringMap &stringMap)
{
    std::string stringData = stringMap.stringData;
    int actual_size = (stringMap.dim1-6)*(stringMap.dim2-6);
    // std::vector<int> results(rowLengthVector.size());
    char *d_data;
    int *d_results;
    CUDA_CHECK(cudaMalloc(&d_data, stringData.size() * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_results, actual_size * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_data, stringData.data(),
                          stringData.size() * sizeof(char),
                          cudaMemcpyHostToDevice));

    int blockSize = 1024;
    int numBlocks = (actual_size + blockSize - 1) / blockSize;
    processStringXMAS<<<numBlocks, blockSize>>>(d_data, d_results, stringMap.dim1, stringMap.dim2);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Process 1 complete" << std::endl;

    int total_sum = sumWithThrust(d_results, actual_size);

    std::cout << "Process 2 complete" << std::endl;
    std::cout << total_sum << std::endl;
    CUDA_CHECK(cudaDeviceSynchronize());

    // CUDA_CHECK(cudaMemcpy(&results[0], d_results, rowLengthVector.size()*sizeof(int), cudaMemcpyDeviceToHost));
    // for (int i: results){
    //     std::cout << i <<" ";
    // }
    // std::cout<<std::endl;
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_results));
}

void puzzle_2(StringMap &stringMap)
{
    std::string stringData = stringMap.stringData;
    int actual_size = (stringMap.dim1-6)*(stringMap.dim2-6);
    // std::vector<int> results(rowLengthVector.size());
    char *d_data;
    int *d_results;
    CUDA_CHECK(cudaMalloc(&d_data, stringData.size() * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_results, actual_size * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_data, stringData.data(),
                          stringData.size() * sizeof(char),
                          cudaMemcpyHostToDevice));

    int blockSize = 1024;
    int numBlocks = (actual_size + blockSize - 1) / blockSize;
    processStringX_MAS<<<numBlocks, blockSize>>>(d_data, d_results, stringMap.dim1, stringMap.dim2);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Process 1 complete" << std::endl;

    int total_sum = sumWithThrust(d_results, actual_size);

    std::cout << "Process 2 complete" << std::endl;
    std::cout << total_sum << std::endl;
    CUDA_CHECK(cudaDeviceSynchronize());

    // CUDA_CHECK(cudaMemcpy(&results[0], d_results, rowLengthVector.size()*sizeof(int), cudaMemcpyDeviceToHost));
    // for (int i: results){
    //     std::cout << i <<" ";
    // }
    // std::cout<<std::endl;
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_results));
}