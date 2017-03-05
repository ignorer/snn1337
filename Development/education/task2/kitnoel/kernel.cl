__kernel void job(__global const int* A, __global const int* B, __global int* C, const int n) {
	int i = get_global_id(0);
	int j = get_global_id(1);
    C[i * n + j] = 0;
    for (int k = 0; k < n; ++k) {
         C[i * n + j] += A[i * n + k] * B[k * n + j];
    }
}