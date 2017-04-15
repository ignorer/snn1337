__kernel void matrixMultiplication(__global int* A, __global int* B, __global int* C,
    int const n, int const between, int const m) {
	int i = get_global_id(0);
	int j = get_global_id(1);
    C[i * m + j] = 0;
    for (int k = 0; k < between; k++) {
        C[i * m + j] += A[i * between + k] * B[k * m + j];
    }
}