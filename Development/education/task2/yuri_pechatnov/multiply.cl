/*!
 * Multiplies raw matrixes
 * @param messageA - First raw matrix
 * @param messageB - Transposed second raw matrix
 * @param messageResult - Field to write raw result matrix
 */
__kernel void kernel_multiply(__global int *messageA,
                              __global int *messageB,
                              __global int *messageResult) {
	// get coord
	int x = get_global_id(0);
	int y = get_global_id(1);
	
	// get first matrix parameters
	int N1 = messageA[0];
	int M1 = messageA[1];
	int offsetA = 2 + x * M1;
	
	// get second matrix parameters
	int M2 = messageB[0];
	int N2 = messageB[1];
	int offsetB = 2 + y * N2;
	
	// set result
	messageResult[0] = N1;
	messageResult[1] = M2;
	
	int result = 0;
	for (int i = 0; i < M1; i++) {
		result = result +
		         messageA[offsetA + i] * messageB[offsetB + i];
	}
	
	messageResult[2 + M2 * x + y] = result;
}