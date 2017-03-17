__kernel void arrayIndeces(__global int* pOutputVector) {
	int gid = get_global_id(0);
	pOutputVector[gid] = gid;
}