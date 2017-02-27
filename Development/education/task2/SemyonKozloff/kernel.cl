
__kernel void multiplier(__global int* a, __global int* b, __global int* с, int n) {
    int k, i, gid;
	gid = get_global_id(0);
	for (k = 0; k < n; ++k) {
        c[gid * n + k] = 0;
        for (i = 0; i < n; ++i) {
            c[gid * n + k] += a[gid * n + i] * b[i * n + k];
        }
    }
}