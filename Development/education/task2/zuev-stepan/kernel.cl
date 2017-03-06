__kernel void SimpleKernel(__global const int* a, __global const int* b, __global int* res, int size)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x > size || y > size)
    {
        return;
    }
    res[x * size + y] = 0;
    int i = 0;
    for (i = 0; i < size; ++i)
    {
        res[x * size + y] += a[x * size + i] * b[i * size + y];
    }
}