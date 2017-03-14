__kernel void kernel_enumerate(__global int *message) {
	int gid = get_global_id(0);
	message[gid] = gid;
}