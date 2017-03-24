#include <vector>
#include <chrono>
#include <iostream>
#include <string>

using namespace std;

using Vector = vector<int>;
using Matrix = vector<Vector>;

void printMatrix(const Matrix& m) {
    for (auto& row : m) {
        for (auto& element : row) {
            cout << element << " ";
        }
        cout << endl;
    }
}

int main() {
    int n = 1000;
    Matrix A(n, Vector(n));
    Matrix B(n, Vector(n));
    Matrix C(n, Vector(n));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
        }
    }

    using namespace chrono;
    auto start = steady_clock::now();

    Matrix B2(n, Vector(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            B2[i][j] = B[j][i];
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            auto aRow = A[i].data();
            auto bRow = B2[j].data();
            int sum = 0;
            for (int l = 0; l < n; ++l) {
                sum += aRow[l] * bRow[l];
            }
            C[i][j] = sum;
        }
    }
    cout << duration_cast<milliseconds>(steady_clock::now() - start).count() << endl;
}

