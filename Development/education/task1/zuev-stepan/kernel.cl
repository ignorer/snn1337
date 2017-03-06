__kernel void SimpleKernel(__global int* a, unsigned int size)
{
    int myId = get_global_id(0);
    if (myId > size)
    {
        return;
    }

    a[myId] = myId;
}