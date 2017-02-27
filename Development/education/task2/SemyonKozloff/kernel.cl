
__kernel void multiplier(__global int* a, __global int* b, __global int* c, unsigned n) {
	int gid = get_global_id(0);
	for (int k = 0; k < n; ++k) {
		c[gid * n + k] = 0;
		for (int i = 0; i < n; ++i) {
			c[gid * n + k] += a[gid * n + i] * b[i * n + k];
		}
	}
}