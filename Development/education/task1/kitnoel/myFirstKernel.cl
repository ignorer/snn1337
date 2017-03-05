__kernel void job(__global int *buffer) {
	int gid = get_global_id(0);
	buffer[gid] = gid;
}