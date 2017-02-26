__kernel void test(__global int* message)
{
    int gid = get_global_id(0);
    message[gid] += gid;
}