__kernel void kernel1(__global int* message) {
	int gid = get_global_id(0);
	message[gid] += gid;
}