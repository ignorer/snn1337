float dotProductLocal(__local const float* a, __global const float* b, size_t size) {
    float res = 0;
    for (size_t i = 0; i < size; i++) {
        res += a[i] * b[i];
    }
    return res;
}

float dotProduct(__global const float* a, __global const float* b, size_t size) {
    float res = 0;
    for (size_t i = 0; i < size; i++) {
        res += a[i] * b[i];
    }
    return res;
}

float activationFunction(float x, int layerId, int layersNumber) {
    return layerId < layersNumber - 2 ?
            1.7159 * tanh(0.66666667 * x) : // hyperbolic tangent for all hidden layers
            1 / (1 + exp(-x)); // sigmoid for output layer
}

__kernel void batchedNeuron(
        // network parameters
        __global const int* layerSizes,
        int layersNumber,
        __global const float* weights,
        // per-batch parameters
        __global float* values,
        int batchSize,
        // per-layer parameters
        int layerId,
        int weightsOffset,
        int valuesOffset
) {
    int neuronId = get_global_id(0);
    int previousLayerSize = layerSizes[layerId];
    int layerSize = layerSizes[layerId + 1];

    __global const float* weightsVector = weights + weightsOffset + (neuronId % layerSize) * (previousLayerSize + 1);
    __global const float* valuesVector = values + valuesOffset;
    int biasCorrection = (layerId < layersNumber - 2) ? 1 : 0;
    int targetIndex = valuesOffset + (previousLayerSize + 1) * batchSize + neuronId + biasCorrection;
    for (int i = 0; i < batchSize; ++i) {
        float resultValue = dotProduct(weightsVector, valuesVector, previousLayerSize + 1);
        values[targetIndex] = activationFunction(resultValue, layerId, layersNumber);
        valuesVector += previousLayerSize + 1;
        targetIndex += layerSize + biasCorrection;
    }
}

__kernel void batchedNeuronParallel(
        // network parameters
        __global const int* layerSizes,
        int layersNumber,
        __global const float* weights,
        // per-batch parameters
        __global float* values,
        int batchSize,
        // per-layer parameters
        int layerId,
        int weightsOffset,
        int valuesOffset,
        __local float* weightsCache
) {
    int neuronId = get_global_id(0);
    int previousLayerSize = layerSizes[layerId];
    int layerSize = layerSizes[layerId + 1];
    int step = get_global_size(0) / layerSize;

    __global const float* weightsVector = weights + weightsOffset + (neuronId % layerSize) * (previousLayerSize + 1);
    for (int i = 0; i < previousLayerSize + 1; ++i) {
        weightsCache[i] = weightsVector[i];
    }
    __global const float* valuesVector = values + valuesOffset + neuronId / layerSize * (previousLayerSize + 1);
    int biasCorrection = (layerId < layersNumber - 2) ? 1 : 0;
    int targetIndex = valuesOffset + (previousLayerSize + 1) * batchSize + neuronId + biasCorrection + neuronId / layerSize * biasCorrection;
    for (int i = 0; i < batchSize / step; ++i) {
        float resultValue = dotProductLocal(weightsCache, valuesVector, previousLayerSize + 1);
        values[targetIndex] = activationFunction(resultValue, layerId, layersNumber);
        valuesVector += (previousLayerSize + 1) * step;
        targetIndex += (layerSize + biasCorrection) * step;
    }
}