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

struct RowInfo {
    int row_length;
    int start_idx;
};

struct FileData {
    std::vector<int> data;
    RowInfo rowInfo;
};

std::vector<FileData> readNumbers(const char* filename) {
    std::vector<FileData> fileDataVector;
    std::ifstream file(filename);
    std::string line;
    int row_start = 0;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        FileData fileData;
        int n;
        while(iss >> n){
            fileData.data.push_back(n);
        }
        fileData.rowInfo.row_length = fileData.data.size();
        fileData.rowInfo.start_idx = row_start;
        row_start += fileData.rowInfo.row_length;
        fileDataVector.push_back(fileData);
    }
    
    return fileDataVector;
}

__global__ void processRows(const int* data, int* rowStarts, int* rowLengths, int* results, int count) {
    // Calculate which pair this thread should process
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < count) {
        results[idx] = 1;
        int rowStart = rowStarts[idx];
        int rowLength = rowLengths[idx];
        int prevDiff = data[rowStart] - data[rowStart+1];
        int currDiff = 0;
        if ((abs(prevDiff)<1) || (abs(prevDiff)>3)){
            results[idx] = 0;
        } 
        else {
            for (int i = 1; i< rowLength-1; i++){
                currDiff = data[rowStart+i] - data[rowStart+i+1];
                if ((abs(currDiff)<1) || (abs(currDiff)>3) || (prevDiff*currDiff<=0)){
                    results[idx] = 0;
                    break;
                }
                prevDiff = currDiff;
            }
        }
    }
}

__global__ void processRowsWithTolerance(const int* data, int* rowStarts, int* rowLengths, int* results, int count) {
    // Calculate which pair this thread should process
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < count) {
        results[idx] = 0;
        int rowStart = rowStarts[idx];
        int rowLength = rowLengths[idx];
        for (int j=0; j < rowLength; j++){
            int result = 1;
            
            int idx_1 = j==0 ? 1: 0;
            int idx_2 = (j!=idx_1 + 1)? idx_1+1: idx_1+2; 

            int prevDiff = data[rowStart+ idx_1] - data[rowStart+idx_2];
            int currDiff = 0;
            if ((abs(prevDiff)<1) || (abs(prevDiff)>3)){
                result = 0;
            } 
            else {
                for (int i = 1; i< rowLength-2; i++){
                    idx_1=idx_2;
                    idx_2=(j!=idx_1 + 1)? idx_1+1: idx_1+2;
                    currDiff = data[rowStart+idx_1] - data[rowStart+idx_2];
                    if ((abs(currDiff)<1) || (abs(currDiff)>3) || (prevDiff*currDiff<=0)){
                        result = 0;
                        break;
                    }
                    prevDiff = currDiff;
                }
            }
            results[idx] = results[idx] || result; 
            if (result==1) break;
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

void puzzle_1(std::vector<int> &dataVector, std::vector<int> &rowStartsVector, std::vector<int> &rowLengthVector);
void puzzle_2(std::vector<int> &dataVector, std::vector<int> &rowStartsVector, std::vector<int> &rowLengthVector);

// void puzzle_2(int pairCount, std::vector<NumberPair> &pairs);

int main()
{

    char cwd[1024];
    chdir("../day02/");
    getcwd(cwd, sizeof(cwd));
    printf("Current working dir: %s\n", cwd);
    std::vector<FileData> fileDataVector = readNumbers("input.txt");
    std::vector<int> dataVector;
    std::vector<int> rowStartsVector;
    std::vector<int> rowLengthVector;
    
    for (FileData fileData: fileDataVector){
        // for (int i: fileData.data){
        //     std::cout << i << ' ';        
        // }
        dataVector.insert(dataVector.end(), fileData.data.begin(), fileData.data.end());
        // std::cout << std::endl;
        // std::cout << fileData.rowInfo.start_idx << std::endl;
        // std::cout << fileData.rowInfo.row_length << std::endl;
        rowStartsVector.push_back(fileData.rowInfo.start_idx);
        rowLengthVector.push_back(fileData.rowInfo.row_length);
    }
    // for (int i: dataVector){
    //     std::cout << i << ' ';        
    // }
    // std::cout<<std::endl;
    puzzle_1(dataVector, rowStartsVector, rowLengthVector);
    puzzle_2(dataVector, rowStartsVector, rowLengthVector);

    return 0;
}

void puzzle_1(std::vector<int> &dataVector, std::vector<int> &rowStartsVector, std::vector<int> &rowLengthVector)
{
    // std::vector<int> results(rowLengthVector.size());
    int *d_data;
    int *d_results;
    int *d_rowStarts;
    int *d_rowLength;
    CUDA_CHECK(cudaMalloc(&d_data, dataVector.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_results, rowLengthVector.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rowStarts, rowStartsVector.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rowLength, rowLengthVector.size() * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_data, dataVector.data(),
                          dataVector.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowStarts, rowStartsVector.data(),
                          rowStartsVector.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowLength, rowLengthVector.data(),
                          rowLengthVector.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    int blockSize = 256;
    int numBlocks = (rowStartsVector.size() + blockSize - 1) / blockSize;
    processRows<<<numBlocks, blockSize>>>(d_data, d_rowStarts, d_rowLength, d_results, rowLengthVector.size());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Process 1 complete" << std::endl;

    int total_sum = sumWithThrust(d_results, rowLengthVector.size());

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
    CUDA_CHECK(cudaFree(d_rowStarts));
    CUDA_CHECK(cudaFree(d_rowLength));
}

void puzzle_2(std::vector<int> &dataVector, std::vector<int> &rowStartsVector, std::vector<int> &rowLengthVector)
{
    // std::vector<int> results(rowLengthVector.size());
    int *d_data;
    int *d_results;
    int *d_rowStarts;
    int *d_rowLength;
    CUDA_CHECK(cudaMalloc(&d_data, dataVector.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_results, rowLengthVector.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rowStarts, rowStartsVector.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rowLength, rowLengthVector.size() * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_data, dataVector.data(),
                          dataVector.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowStarts, rowStartsVector.data(),
                          rowStartsVector.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowLength, rowLengthVector.data(),
                          rowLengthVector.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    int blockSize = 256;
    int numBlocks = (rowStartsVector.size() + blockSize - 1) / blockSize;
    processRowsWithTolerance<<<numBlocks, blockSize>>>(d_data, d_rowStarts, d_rowLength, d_results, rowLengthVector.size());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Process 1 complete" << std::endl;

    int total_sum = sumWithThrust(d_results, rowLengthVector.size());

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
    CUDA_CHECK(cudaFree(d_rowStarts));
    CUDA_CHECK(cudaFree(d_rowLength));
}
// void puzzle_2(int pairCount, std::vector<NumberPair> &pairs)
// {
//     NumberPair *d_pairs;
//     int *d_results;
//     int *d_firsts;
//     int *d_seconds;
//     CUDA_CHECK(cudaMalloc(&d_pairs, pairCount * sizeof(NumberPair)));
//     CUDA_CHECK(cudaMalloc(&d_results, pairCount * sizeof(int)));
//     CUDA_CHECK(cudaMalloc(&d_firsts, pairCount * sizeof(int)));
//     CUDA_CHECK(cudaMalloc(&d_seconds, pairCount * sizeof(int)));

//     CUDA_CHECK(cudaMemcpy(d_pairs, pairs.data(),
//                           pairCount * sizeof(NumberPair),
//                           cudaMemcpyHostToDevice));
//     int blockSize = 256;
//     int numBlocks = (pairCount + blockSize - 1) / blockSize;
//     processPairs<<<numBlocks, blockSize>>>(d_pairs, d_firsts, d_seconds, pairCount);
//     CUDA_CHECK(cudaDeviceSynchronize());
//     std::cout << "Process 1 complete" << std::endl;


//     findAndAddDuplicate<<<numBlocks, blockSize>>>(d_firsts, d_seconds, d_results, pairCount);
//     CUDA_CHECK(cudaDeviceSynchronize());

//     std::cout << "Process 3 complete" << std::endl;
//     int total_sum = sumWithThrust(d_results, pairCount);

//     std::cout << "Process 4 complete" << std::endl;
//     std::cout << total_sum << std::endl;

//     CUDA_CHECK(cudaFree(d_pairs));
//     CUDA_CHECK(cudaFree(d_results));
//     CUDA_CHECK(cudaFree(d_firsts));
//     CUDA_CHECK(cudaFree(d_seconds));
// }