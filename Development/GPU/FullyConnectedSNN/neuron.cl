#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#define DELTA_T 15
#define SPIKES_SIZE 2048

inline float refractoryFunc(float t, float threshold, float tau) { // функция восстановления
    if (t >= 0)
        return -threshold * native_exp(1 - 2 * t / tau);
    return 0;
}

inline float spikeFunc(float t, float tau) { // функция распада
    if (t >= 0)
        return t / tau * native_exp(1 - t / tau);
    return 0;
}

void calcRecvPot(__private float* recvPot,__global const float* weights, __global const int* spikes, // часть работы кернеля
                 int spikesPerSyn, int synPerConn, size_t connNum) {
    // recivedPot is a precalculated hash-table [{spikeTime : its potential}, ...]
    int i;
    int j;
    for (j = 0; j < connNum * synPerConn; ++j) {
        for (i = 0; i < spikesPerSyn; ++i)
            recvPot[spikes[j * spikesPerSyn + i]] += weights[j] * spikeFunc(spikes[j * spikesPerSyn + i], 1.0);
    }
}

int calcSpikesNum(__global const int* layersSz, int layerNum, int synPerConn, int spikesPerSyn) {
    int result = 0;
    int i;
    for(i = 0; i < layerNum; ++i) {
        result += layersSz[i] * synPerConn * spikesPerSyn;
    }
    return result;
}

int calcLayerId(__global const int* layersSz, int globalId) { // globalId - количество вычислителей
    int result = 0;
    do {
        globalId -= layersSz[++result];
    } while (globalId >= 0);
    return result;
}

int calcNeuronId(__global const int* layersSz, int globalId, size_t layerId) {
    int i;
    for (i = 0; i < layerId; ++i) { //я надеюсь globalID считается с нуля
        globalId -= layersSz[i];
    }
    return globalId;
}

__global int* calcPrevLayerStart(__global const float* input,
                                 __global int* spikes,
                                 __global const int* layersSz,
                                 int layerId,
                                 const int synPerConn,
                                 const int spikesPerSyn) {
    if (layerId == 0) {
        int i;
        int j = 0;
        size_t encInputSize = layersSz[0] * synPerConn * spikesPerSyn;
        int freq = 0;
        for (i = 0; i < encInputSize; i++) {
            freq += input[j++];
            if (freq / 1000 > 0) {
                spikes[i++] = j;
                freq = freq % 1000;
            }
        }
        return spikes;
    }
    int i;
    for (i = 0; i < layerId - 1; ++i) {
        spikes += layersSz[i] * synPerConn * spikesPerSyn;
    }
    return spikes;
}

__global const float* calcWeightsStart(__global const float* weights,
                                       __global const int* layersSz,
                                       const int synPerConn,
                                       int layerId,
                                       int neuronId) {
    int i;
    // skip all Prev layers
    for (i = 0; i < layerId - 1; ++i) {
        weights += layersSz[i] * layersSz[i + 1] * synPerConn;
    }
    // skip all neurons on this layer
    weights += (layersSz[layerId - 1]) * synPerConn * neuronId;
    return weights;
}

__global int* calcTargetSpikesPtr(__global int* spikes,
                                  __global int* output,
                                  const int layersNum,
                                  const int spikesNum,
                                  const int layerId,
                                  const int globalId) {
    if (layerId < layersNum - 1) { // if we're on hidden layer
        __global int* result = spikes + globalId;
        return result;
    }
    return output + (globalId - spikesNum); // if we're on output layer
}

__kernel void neuron(
        __global const int* layersSz,
        const int layersNum,
        int synPerConn,
        int spikesPerSyn,
        int exitTime,
        __global const float* weights,
        __global int* spikes,
        float threshold,
        volatile __global int* t,
        __global int* sem,
        __global const float* input,
        __global int* output
) {
    int globalId = get_global_id(0);
    int layerId = calcLayerId(layersSz, globalId);
    int neuronId = calcNeuronId(layersSz, globalId, layerId);
    int spikesNum = calcSpikesNum(layersSz, layersNum, synPerConn, spikesPerSyn);
    __global int* outputVec = calcTargetSpikesPtr(spikes, output, layersNum, spikesNum, layerId, globalId);
    __global const int* inputSpikesVec = calcPrevLayerStart(input, spikes, layersSz, layerId, synPerConn, spikesPerSyn);
    __global const float* weightsVec = calcWeightsStart(weights, layersSz, synPerConn, layerId, neuronId);
    __private float spikePots[SPIKES_SIZE]; // FIX ME: should be __global argument
    int i;
    for(i = 0; i < SPIKES_SIZE; ++i)
        spikePots[i] = 0.0;
    calcRecvPot(spikePots, weightsVec, inputSpikesVec, spikesPerSyn, synPerConn, layersSz[layerId - 1]);

    int j;
    float pot = 0;
    while (*t < exitTime) {
        for (i = 0; i < DELTA_T; ++i) {
            pot += spikePots[*t+i]; // could be no spike in time (*t + i)
            if (pot > threshold) {
                outputVec[j++] = *t + i;
                pot += refractoryFunc(i, threshold, 1.0); // i is current spike's time = (*t+i) - *t
            }
        }
        atomic_cmpxchg(t, *t + DELTA_T, *t + DELTA_T);
    }
}