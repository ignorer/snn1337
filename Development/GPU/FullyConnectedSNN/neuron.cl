#define DELTA_T 15
#define SPIKES_SIZE 2048
#define SPIKE_FUNCTION_PARAM 2.7
#define  REFRACTORY_FUNCTION_PARAM 20.0

inline float refractoryFunc(float t, float threshold, float tau) {
    if (t >= 0)
        return -threshold * native_exp(1 - 2 * t / tau);
    return 0;
}

inline float spikeFunc(float t, float tau) {
    if (t >= 0)
        return t / tau * native_exp(1 - t / tau);
    return 0;
}

void calcRecvPot(__private float* recvPot,__global const float* weights, __global const int* spikes,
  int spikesPerSyn, int synPerConn, size_t connNum) {
    // recievedPot is a precalculated hash-table [{spikeId: its potential}, ...]
    int i;
    int j;
    for (j = 0; j < connNum * synPerConn; ++j) {
        for (i = 0; i < spikesPerSyn; ++i)
            recvPot[i] += weights[j] * spikeFunc(spikes[j * spikesPerSyn + i], SPIKE_FUNCTION_PARAM);
    }
}

int calcSpikesNum(__global const int* layerSizes, int layerNum, int synPerConn, int spikesPerSyn) {
    int result = 0;
    int i;
    for(i = 0; i < layerNum; ++i) {
        result += layerSizes[i] * synPerConn * spikesPerSyn;
    }
    return result;
}

int calcLayerId(__global const int* layerSizes, int globalId) {
    int result = 0;
    do {
        globalId -= layerSizes[++result];
    } while (globalId >= 0);
    return result;
}

int calcNeuronId(__global const int* layerSizes, int globalId, size_t layerId) {
    int i;
    for (i = 0; i < layerId; ++i) { // i = 1 or  i = 0??
        globalId -= layerSizes[i];
    }
    return globalId;
}

__global int* calcPrevLayerStart(__global const float* input, __global int* spikes,
  __global const int* layerSizes, int layerId, const int synPerConn, const int spikesPerSyn) {
    if (layerId == 1) {
        int i;
        int j;
        size_t encInputSize = layerSizes[0] * synPerConn * spikesPerSyn;
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
    for (i = 1; i < layerId - 1; ++i) {
        spikes += layerSizes[i] * synPerConn * spikesPerSyn;
    }
    return spikes;
}

__global const float* calcWeightsStart(__global const float* weights, __global const int* layerSizes,
  const int synPerConn, int layerId, int neuronId) {
    int i;
    // skip all Prev layers
    for (i = 0; i < layerId - 1; ++i) {
        weights += layerSizes[i] * layerSizes[i + 1] * synPerConn;
    }
    // skip all neurons on this layer
    weights += (layerSizes[layerId - 1]) * synPerConn * neuronId;
    return weights;
}

__global int* calcTargetSpikesPtr(__global int* spikes, __global int* output,
  const int layersNum, const int spikesNum, const int layerId, const int globalId) {
    if (layerId < layersNum - 1) { // if we're on hidden layer
        __global int* result = spikes + globalId;
        return result;
    }
    return output + (globalId - spikesNum); // if we're on output layer
}

__kernel void neuron(
        __global const int* layerSizes,
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
    int layerId = calcLayerId(layerSizes, globalId);
    int neuronId = calcNeuronId(layerSizes, globalId, layerId);
    if (globalId == 0)
        *t = 0;

    int spikesNum = calcSpikesNum(layerSizes, layersNum, synPerConn, spikesPerSyn);

    // spikes of current neuron
    __global int* outputVec = calcTargetSpikesPtr(spikes, output, layersNum, spikesNum, layerId, globalId);

    // spikes of previous neuron
    __global const int* inputSpikesVec = calcPrevLayerStart(input, spikes, layerSizes, layerId, synPerConn, spikesPerSyn);

    __global const float* weightsVec = calcWeightsStart(weights, layerSizes, synPerConn, layerId, neuronId);

    // precalculate potentials that produce each recieved spike
    // on same layer this rule is always correct:
    // i < j ==> spikes[i] < spikes[j] ==> pot. of spikes[i] is spikePot[i]
    __private float spikePots[SPIKES_SIZE];
    int i;
    for(i = 0; i < SPIKES_SIZE; ++i)
        spikePots[i] = 0.0;
    calcRecvPot(spikePots, weightsVec, inputSpikesVec, spikesPerSyn, synPerConn, layerSizes[layerId - 1]);

    int j = 0;
    int k = 0;
    float pot = 0;
    while (*t < exitTime) {
        for (i = 0; i < DELTA_T; ++i) {
            if (k < spikesPerSyn * synPerConn &&  spikePots[k] <= *t + i)
                pot += spikePots[k++];
            if (pot >= threshold) {
                if ( (layerId == layersNum - 1 && j < exitTime * layerSizes[layerId])
                || (layerId < layersNum - 1 && j < layerSizes[layerId + 1]))
                outputVec[j++] = *t + i;
                pot += refractoryFunc(*t + i - k, threshold, REFRACTORY_FUNCTION_PARAM);

            }
        }
        if (globalId == 0)
            *t += DELTA_T;
    }
}