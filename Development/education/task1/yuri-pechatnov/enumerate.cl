__kernel void kernel_enumerate(__global int *message) {
	message[gid] = get_global_id(0);
}