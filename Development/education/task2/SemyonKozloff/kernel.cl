
__kernel void multiplier(__global int* a, __global int* b, __global int* c, __constant int* n) {
	int dim = n[0];
	int gid = get_global_id(0);
	for (int k = 0; k < dim; ++k) {
        c[gid * dim + k] = 0;
        for (int i = 0; i < dim; ++i) {
            c[gid * dim + k] += a[gid * dim + i] * b[i * dim + k];
        }
    }
}