#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cmath>

using namespace std;

template<int BLOCK_SIZE>
__global__ void matmul_kernel(const int* A, const int* B, int* C, int N) {
    __shared__ int sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int sB[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    int sum = 0;
    for (int k = 0; k < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++k) {
        if (row < N && k * BLOCK_SIZE + tx < N)
            sA[ty][tx] = A[row * N + k * BLOCK_SIZE + tx];
        else
            sA[ty][tx] = 0;

        if (col < N && k * BLOCK_SIZE + ty < N)
            sB[ty][tx] = B[(k * BLOCK_SIZE + ty) * N + col];
        else
            sB[ty][tx] = 0;

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += sA[ty][i] * sB[i][tx];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

vector<int> readMatrix(const string& filename, int& N) {
    ifstream f(filename);
    f >> N;
    vector<int> mat(N * N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            f >> mat[i * N + j];
    return mat;
}

void writeMatrix(const string& filename, const vector<int>& mat, int N) {
    ofstream f(filename);
    f << N << "\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            f << mat[i * N + j] << " ";
        f << "\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <block_size>" << endl;
        return 1;
    }
    int block_size = atoi(argv[1]);
    if (block_size != 8 && block_size != 16 && block_size != 32) {
        cerr << "Block size must be 8, 16 or 32" << endl;
        return 1;
    }

    int N1, N2;
    auto A_host = readMatrix("data/matrix_a.txt", N1);
    auto B_host = readMatrix("data/matrix_b.txt", N2);
    if (N1 != N2) {
        cerr << "Matrix sizes do not match" << endl;
        return 1;
    }
    int N = N1;

    int *A_dev = nullptr, *B_dev = nullptr, *C_dev = nullptr;
    size_t bytes = N * N * sizeof(int);
    cudaMalloc(&A_dev, bytes);
    cudaMalloc(&B_dev, bytes);
    cudaMalloc(&C_dev, bytes);

    cudaMemcpy(A_dev, A_host.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B_host.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threads(block_size, block_size);
    dim3 blocks((N + block_size - 1) / block_size,
                (N + block_size - 1) / block_size);

    auto start = chrono::steady_clock::now();

    switch (block_size) {
        case 8:
            matmul_kernel<8><<<blocks, threads>>>(A_dev, B_dev, C_dev, N);
            break;
        case 16:
            matmul_kernel<16><<<blocks, threads>>>(A_dev, B_dev, C_dev, N);
            break;
        case 32:
            matmul_kernel<32><<<blocks, threads>>>(A_dev, B_dev, C_dev, N);
            break;
    }
    cudaDeviceSynchronize();

    vector<int> C_host(N * N);
    cudaMemcpy(C_host.data(), C_dev, bytes, cudaMemcpyDeviceToHost);

    auto end = chrono::steady_clock::now();
    double elapsed = chrono::duration<double>(end - start).count();

    writeMatrix("data/result_cpp.txt", C_host, N);

    cout << "Execution time: " << elapsed << " seconds" << endl;

    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);

    return 0;
}