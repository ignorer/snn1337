__kernel void MatrixMult(__global int* A, __global int* B, __global int* C, const unsigned int size){
    int col;
    int index;
    int row = get_global_id(0);
    __global int* aRow = A + row*size;

    for(col = 0; col < size; ++col){
        int sum = 0;
        __global int* bRow = B + col * size;
        for(index = 0; index < size; ++index){
            sum += aRow[index]*bRow[index];
        }
        C[row*size +col] = sum;

     }
}