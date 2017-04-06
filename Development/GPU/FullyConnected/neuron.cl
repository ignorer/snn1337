#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

inline float dotProduct(__global const float* a,  __global const float* b, size_t size) {
    float res = 0;
    for (size_t i = 0; i < size; i++) {
        res += a[i]*b[i];
    }
    return res;
}

inline float sigmoid(float x) {
    return 1.7159*tanh(0.66666667*x);
}

int calculateLayerId(__global const int* layerSizes, int globalId) {
    int result = 0;
    do {
        globalId -= layerSizes[++result];
    } while (globalId >= 0);
    return result;
}

int calculateNeuronId(__global const int* layerSizes, int globalId, size_t layerId) {
    int i;
    for (i = 1; i < layerId; ++i) {
        globalId -= layerSizes[i];
    }
    return globalId;
}

int calculateValueId(int globalId, size_t layerId) {
    return globalId + layerId - 1;
}

__global const float* calculatePreviousLayerBeginning(__global const float* input, __global const float* values,
        __global const int* layerSizes, int layerId) {
    int i;
    if (layerId == 1) {
        return input;
    }
    for (i = 1; i < layerId - 1; ++i) {
        values += layerSizes[i];
    }
    return values;
}

__global const float* calculateWeightsVectorBeginning(__global const float* weights, __global const int* layerSizes,
        int layerId, int neuronId) {
    int i;
    // skip all previous layers
    for (i = 0; i < layerId - 1; ++i) {
        weights += (layerSizes[i] + 1) * layerSizes[i + 1]; // +1 because of bias
    }

    // skip all neurons on this layer
    weights += (layerSizes[layerId - 1] + 1) * neuronId;
    return weights;
}

__global float* calculateTargetValuePtr(__global float* values, __global float* output, int valuesNumber,
        int layersNumber, int layerId, int globalId) {
    __global float* result = values + globalId + layerId; // layerId is a number of biases on previous layers
    if (layerId < layersNumber - 1) { // if we're on hidden layer
        return result;
    }
    return output + (result - values - valuesNumber) - 1; // if we're on output layer. -1 because there's no bias on output
}

__kernel void neuron(
        __global const int* layerSizes,
        __global const float* weights,
        __global volatile int* counters,
        __global float* values,
        __global const float* inputs,
        __global float* outputs,
        int valuesNumber,
        int layersNumber) {

    int globalId = get_global_id(0);

    int layerId = calculateLayerId(layerSizes, globalId);
    int neuronId = calculateNeuronId(layerSizes, globalId, layerId);

    __global const float* weightsVector = calculateWeightsVectorBeginning(weights, layerSizes, layerId, neuronId);
    __global const float* valuesVector = calculatePreviousLayerBeginning(inputs, values, layerSizes, layerId);

    while (counters[layerId - 1] > 0) {
//      busy wait
    }

    float resultValue = dotProduct(weightsVector, valuesVector, layerSizes[layerId - 1] + 1);
    *calculateTargetValuePtr(values, outputs, valuesNumber, layersNumber, layerId, globalId) = sigmoid(resultValue);

    atomic_dec(counters + layerId);
}