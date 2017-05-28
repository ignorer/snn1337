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

typedef struct {
    int id;
    int size;
    __global int* spikes;
    __global float* spikePots;
} Layer;

typedef struct {
     int id;
     __global float* weights;
     __global int* spikes; // target
     int firedSpikesNumber;
     float potential;
     Layer* prevLayer;
} Neuron;

typedef struct {
    const int spikesPerSyn;
    const int synPerConn;
    const __global int* layerSizes;
    const int layersNumber;
    int spikesNumber;
    __global int* spikes;
    float threshold;
    const __global float* weights;
    __global int* input;
    __global int* output;
    int exitTime;
} Network;



int calcSpikesNum(Network* net) {
    int result = 0;
    int i;
    for(i = 0; i < net->layersNumber; ++i) {
        result +=net->layerSizes[i] * net->spikesPerSyn;
    }
    return result;
}

int calcLayerId(__global const int* layerSizes, int globalId) {
    int layers = 0;
    int result = 0;
    while (layers < globalId){
        layers += layerSizes[result++];
    }
    return result;
}

int calcNeuronId(__global const int* layerSizes, int globalId, size_t layerId) {
    int i;
    for (i = 0; i < layerId; ++i) {
        globalId -= layerSizes[i];
    }
    return globalId;
}

void calcLayerSpikesPtr(Network* net, Layer* layer) {
    if (layer->id == -1) { // preinput
        layer->spikes = net->input;
    }
    else if (layer->id < net->layersNumber - 1) { // hidden
        int i;
        int prevLayerSizes = 0;
        for (i = 0; i < layer->id; ++i)
            prevLayerSizes += net->layerSizes[i];
        layer->spikes = net->spikes + prevLayerSizes;
    }
    else {
        layer->spikes = net->output;  // output
    }
}

void calcSpikePotsForNeuron(Network* net, Neuron* neuron) {
    int i;
    int j;
    if (neuron->prevLayer->id != -1) {
        int allSynNumber = neuron->prevLayer->size * net->synPerConn;
        for (j = 0; j < allSynNumber; ++j) {
            for (i = 0; i < net->spikesPerSyn; ++i) {
                __global float* spikePots = &neuron->prevLayer->spikePots[j * net->spikesPerSyn + i];
                int spike = neuron->prevLayer->spikes[j * net->spikesPerSyn + i];
                if (spike != -1) //if spike has been fired
                    *spikePots += neuron->weights[j] * spikeFunc(spike, SPIKE_FUNCTION_PARAM);
            }
        }
    }
}

void calcWeightsForNeuron(Network* net, Neuron* neuron) {
    int i;
    // skip all Prev layers
    neuron->weights = (__global float*)net->weights;
    for (i = 0; i < neuron->prevLayer->id - 1; ++i) {
        neuron->weights += net->layerSizes[i] * net->layerSizes[i + 1] * net->synPerConn;
    }
    // skip all neurons on this layer
    neuron->weights += neuron->prevLayer->size * neuron->id * net->synPerConn;
}


void fire(Network* net, Neuron* neuron, int time) {
    int i;
    if (neuron->prevLayer->id == -1) { // input
        size_t encInputSize = net->layerSizes[0] * net->synPerConn * net->spikesPerSyn;
        int freq = 0;
        int t = 0;
        while(t < encInputSize && *net->input > 0) {
            freq += *net->input;
            net->input += 1;
            if (freq / 1000 > 0)
                neuron->spikes[t++] = i;
            freq = freq % 1000;
            i++;
        }
        return;
    }
    for (i = 0; i < neuron->prevLayer->size * net->synPerConn * net->spikesPerSyn; ++i) {
        int inputSpike = neuron->prevLayer->spikes[i];
        if (inputSpike <= time && inputSpike != -1) // -1 stands for unfired yet spike
            neuron->potential += neuron->prevLayer->spikePots[i];
        if (neuron->potential >= net->threshold) {
            neuron->firedSpikesNumber += 1;
            if (!(neuron->prevLayer->id != net->layersNumber - 2 && neuron->firedSpikesNumber > net->spikesPerSyn)) { // calc only first 5 spikes
                *neuron->spikes = time;
                neuron->spikes += 1;
                neuron->potential += refractoryFunc(time - inputSpike, net->threshold, REFRACTORY_FUNCTION_PARAM);
            }
        }
    }
}

void decodeOutput(Network* net) {
    int maxSpikeTrainSize = 0;
    int curSpikeTrainSize = 0;
    int i;
    for (i = 0; i < net->exitTime * net->layerSizes[net->layersNumber-1]; i++) {
       if (net->output[i] == -1) {
           if (curSpikeTrainSize > maxSpikeTrainSize) {
               maxSpikeTrainSize = curSpikeTrainSize;
           }
           curSpikeTrainSize = 0;
       } else
           curSpikeTrainSize += 1;
   }
}

__kernel void neuron(
        __global const int* layerSizes,
        const int layersNum,
        int synPerConn,
        int spikesPerSyn,
        int exitTime,
        __global const float* weights,
        __global int* spikes,
        __global float* spikePotentials,
        float threshold,
        volatile __global int* t,
        __global int* sem,
        __global const int* input,
        __global int* output
) {
    int globalId = get_global_id(0);
    int layerId = calcLayerId(layerSizes, globalId);
    int neuronId = calcNeuronId(layerSizes, globalId, layerId);

    if (globalId == 0)
        *t = 0;

    Network net = {
        .spikesPerSyn = spikesPerSyn,
        .synPerConn = synPerConn,
        .layerSizes = layerSizes,
        .layersNumber = layersNum,
        .spikes = spikes,
        .threshold = threshold,
        .weights = weights,
        .input = (__global int*)input,
        .output = output,
        .exitTime = exitTime
    };
    net.spikesNumber = calcSpikesNum(&net);

    Layer layer = {
        .id = layerId,
        .size = net.layerSizes[layerId],
    };
    calcLayerSpikesPtr(&net, &layer);

    Layer prevLayer = {
        .id = layerId - 1,
        .size = (layerId == 0) ? 0 : layerSizes[layerId - 1],
    };
    calcLayerSpikesPtr(&net, &prevLayer);
    prevLayer.spikePots = spikePotentials + (int)(prevLayer.spikes - spikes);

    Neuron neuron = {
        .id = neuronId,
        .prevLayer = &prevLayer,
        .potential = 0,
        .firedSpikesNumber = 0
    };
    calcWeightsForNeuron(&net, &neuron);
    if (layer.id != net.layersNumber - 1)
        neuron.spikes = layer.spikes + neuronId * spikesPerSyn;
    else
        neuron.spikes = layer.spikes + neuron.id * exitTime;

    int i = 0;
    float pot = 0;
    while (*t < exitTime) {
        calcSpikePotsForNeuron(&net, &neuron);
        for (i = 0; i < DELTA_T; ++i) {
            int now = *t + i;
            fire(&net, &neuron, now);
        }
        if (globalId == 0)
            *t += DELTA_T;
    }
}