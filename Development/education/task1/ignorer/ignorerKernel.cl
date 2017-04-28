void __kernel fillArray(__global int* array) {
    int i = get_global_id(0);
    array[i] = i;
}