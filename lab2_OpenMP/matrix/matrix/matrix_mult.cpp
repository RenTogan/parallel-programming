#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <cstdlib>

using namespace std;
using namespace chrono;

vector<vector<int>> readMatrix(const string& filename, int& n) {
    ifstream file(filename);
    file >> n;
    vector<vector<int>> matrix(n, vector<int>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            file >> matrix[i][j];
    return matrix;
}

void writeMatrix(const string& filename, const vector<vector<int>>& matrix) {
    ofstream file(filename);
    int n = matrix.size();
    file << n << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            file << matrix[i][j] << " ";
        file << endl;
    }
}

int main(int argc, char* argv[]) {
    int num_threads = 4;
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }

    omp_set_num_threads(num_threads);

    int n1, n2;
    auto A = readMatrix("data/matrix_a.txt", n1);
    auto B = readMatrix("data/matrix_b.txt", n2);

    if (n1 != n2) {
        cerr << "Matrices must be same size" << endl;
        return 1;
    }

    int n = n1;
    vector<vector<int>> C(n, vector<int>(n, 0));

    auto start = steady_clock::now();

#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    auto end = steady_clock::now();
    duration<double> duration = end - start;

    writeMatrix("data/result_cpp.txt", C);

    cout << "Threads used: " << num_threads << endl;
    cout << "Matrix size: " << n << "x" << n << endl;
    cout << "Execution time: " << duration.count() << " seconds" << endl;

    return 0;
}