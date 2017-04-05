inline float dotProduct(const __global float* a, size_t aOffset, const __global float* b, size_t bOffset, size_t size) {
    float res = 0;
    for (size_t i = 0; i < size; i++) {
        res += a[aOffset + i]*b[bOffset + i];
    }
    return res;
}

inline float sigmoid(float x) {
    return 1.7159*tanh(0.66666667*x);
}

size_t getNeuronGid(const global int* layerSizes, size_t layerId, size_t local_neuronId) {
    size_t global_neuronId = 0;
    for(size_t i=0; i < layerId; i++) {
        global_neuronId += layerSizes[i];
    }
    global_neuronId += local_neuronId;
    return global_neuronId;
}

__kernel void neuron(
    __global const int* layerSizes,
    __global const float* weights,
    __global int* counters,
    __global float* values,
    __global const float* inputs,
    __global float* outputs) {

    int neuronId = get_local_id(0);
    int layerId = get_group_id(0);
    int numBiases = layerId + 1;
    size_t weightsOffset = getNeuronGid(layerSizes, layerId, neuronId) + numBiases;
    if (layerId == 0) {
        values[neuronId + numBiases] = sigmoid(dotProduct(weights, weightsOffset, inputs, 0, layerSizes[layerId]));
    }
    else {
        size_t valuesOffset = getNeuronGid(layerSizes, layerId-1, 0);
        if (layerId == get_num_groups(0)-1)
            outputs[neuronId] =  sigmoid(dotProduct(weights, weightsOffset, values, valuesOffset, layerSizes[layerId]));
        else {
            size_t valuesId = getNeuronGid(layerSizes, layerId, neuronId) + numBiases;
            values[valuesId] = sigmoid(dotProduct(weights, weightsOffset, values, valuesOffset, layerSizes[layerId]));
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}