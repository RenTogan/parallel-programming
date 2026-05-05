#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

vector<int> generateRandomMatrix(int n) {
    vector<int> matrix(n * n);
    for (int i = 0; i < n * n; i++) {
        matrix[i] = rand() % 10;
    }
    return matrix;
}

void writeMatrixFlat(const string& filename, const vector<int>& matrix, int n) {
    ofstream file(filename.c_str());
    file << n << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            file << matrix[i * n + j] << " ";
        file << endl;
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = atoi(argv[1]);
    vector<int> A, B, C;

    if (rank == 0) {
        srand(time(NULL));
        A = generateRandomMatrix(n);
        B = generateRandomMatrix(n);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rows_per_process = n / size;
    vector<int> local_A(rows_per_process * n);
    vector<int> local_C(rows_per_process * n, 0);

    if (rank != 0) B.resize(n * n);

    double start_time = MPI_Wtime();

    MPI_Bcast(B.data(), n * n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(A.data(), rows_per_process * n, MPI_INT,
        local_A.data(), rows_per_process * n, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < rows_per_process; i++) {
        for (int j = 0; j < n; j++) {
            int sum = 0;
            for (int k = 0; k < n; k++) {
                sum += local_A[i * n + k] * B[k * n + j];
            }
            local_C[i * n + j] = sum;
        }
    }

    if (rank == 0) C.resize(n * n);
    MPI_Gather(local_C.data(), rows_per_process * n, MPI_INT,
        C.data(), rows_per_process * n, MPI_INT, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    if (rank == 0) {
        writeMatrixFlat("data/result_cpp.txt", C, n);
        cout << "Processes used: " << size << endl;
        cout << "Matrix size: " << n << "x" << n << endl;
        cout << "Execution time: " << end_time - start_time << " seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}