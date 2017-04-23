#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#define DELTA_T 15

inline float dotProduct(__global const float* a,  __global const float* b, size_t size) {
    float res = 0;
    for (size_t i = 0; i < size; ++i) {
        res += a[i]*b[i];
    }
    return res;
}

inline float refractoryFunc(float t, float threshold, float tau) {
    if (t >= 0)
        return -threshold * native_exp(1 - 2 * t / tau);
    return 0;
}

inline float spikeFunc(float t, float tau) {
    if (t >= 0)
        return 2 / tau * native_exp(1 - t / tau);
    return 0;
}

inline __local float* spikeFuncVect(__global int* times, size_t size, float tau) {
     int i;
     /* FIX FIX FIX*/
     __local float potentials[1000];
     for (i = 0; i < size; ++i) {
         potentials[i] = spikeFunc(times[i], tau);
     }
     return potentials;
}

int calculateSpikesNumber(__global const int* layersSizes, int layerNumber, int synapsesPerConnection, int spikesPerSynapse) {
    int result = 0;
    int i;
    for(i = 0; i < layerNumber; ++i) {
         result += layersSizes[i] * synapsesPerConnection * spikesPerSynapse;
    }
    return result;
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
    for (i = 1; i < layerId; ++i) { // i = 1 or  i = 0??
        globalId -= layerSizes[i];
    }
    return globalId;
}

__global const float* calculatePreviousLayerBeginning(__global const float* input, __global const int* spikes,
        __global const int* layerSizes, int layerId, const int synapsesPerConnection, const int spikesPerSynapse) {
    if (layerId == 1) {
        int i;
        int j;
        /* FIX FIX FIX */
        __local int inputSpikes[5 * 100];
        size_t inputSize = 5 * 100;
        int t = 0;
        for (i = 0; i < inputSize; i++) {
            t += input[i];
            if (t / 1000 > 0) {
                inputSpikes[j++] = i;
                t = t % 1000;
            }
        }
        return inputSpikes;
    }
    int i;
    for (i = 1; i < layerId - 1; ++i) {
        spikes += layerSizes[i] * synapsesPerConnection * spikesPerSynapse;
    }
    return spikes;
}

__global const float* calculateWeightsBeginning(__global const float* weights, __global const int* layerSizes,
        const int synapsesPerConnection, const int spikesPerSynapse, int layerId, int neuronId) {
    int i;
    // skip all previous layers
    for (i = 0; i < layerId - 1; ++i) {
        weights += layerSizes[i] * layerSizes[i + 1] * synapsesPerConnection * spikesPerSynapse;
    }

    // skip all neurons on this layer
    weights += (layerSizes[layerId - 1] + 1) * synapsesPerConnection * spikesPerSynapse * neuronId;
    return weights;
}

__global float* calculateTargetSpikesPtr(__global int* spikes, __global int* output,
        const int layersNum, const int spikesNum, const int layerId, const int globalId) {
    if (layerId < layersNum - 1) { // if we're on hidden layer
        __global float* result = spikes + globalId;
        return result;
    }
    /* FIX ME */
    return output + (globalId - spikesNum); // if we're on output layer
}

__kernel void neuron(
        __global const int* layersSizes,
        const int layersNumber,
        int synapsesPerConnection,
        int spikesPerSynapse,
        int exitTime,
        __global const float* weights,
        __global const int* spikes,
        float threshold,
        __global int* t,
        __global int* sem,
        __global const float* input,
        __global float* output
        ) {
    int globalId = get_global_id(0);
    int layerId = calculateLayerId(layersSizes, globalId);
    int neuronId = calculateNeuronId(layersSizes, globalId, layerId);
    int spikesNumber = calculateSpikesNumber(layersSizes, layersNumber, synapsesPerConnection, spikesPerSynapse);
    __global int* outputVector = calculateTargetSpikesPtr(spikes, output, layersNumber, spikesNumber, layerId, globalId);
    __global const int* inputSpikes = calculatePreviousLayerBeginning(input, spikes, layersSizes, layerId, synapsesPerConnection, spikesPerSynapse);
    int i;
    int j;
    float potential = 0;
    for (i = 0; i < DELTA_T; ++i) {
        __global const float weightsVector = calculateWeightsBeginning(weights, layersSizes, synapsesPerConnection, spikesPerSynapse, layerId, neuronId);
        potential += dotProduct(weightsVector, spikeFuncVect(inputSpikes, inputSpikesSize, 1));

        if (potential > threshold) {
            outputVector[j++] = t + i;
            potential += refractoryFunc(i, threshold, 1.0);
        }
    }

    //TODO synchronization
}