#include "neuron.cl"

int refractoryFunctionTest(){
    if(refractoryFunc(2.3, 0.5, 1.2) < 0){
       return 1;
    }
    else{
        return 0;
    }
}

int spikeFunctionTest(){
    if(spikeFunc(1.3, 2.3) > 0){
        return 1;
    }
    else{
        return 0;
    }
}

int calculateSpikesNumberTest(__global const int* layersSz){
    if(calcSpikesNum(layersSz, 1, 2, 5) == 20){
        return 1;
    }
    else{
        return 0;
    }
}

int calculateLayerIdTest(__global const int* layersSz){
    int flag = 0;
    if(calcLayerId(layersSz, 4) == 2){
        flag = 1;
    }
    else{
        return 0;
    }
    if(calcLayerId(layersSz, 7) == 3){
        flag = 1;
    }
    else{
        return 0;
    }
    if(calcLayerId(layersSz, 1) == 1){
        flag = 1;
    }
    else{
        return 0;
    }
    return flag;
}

int calculateNeuronIdTest(__global const int* layersSz){
//если отсчет идет с 0 то layerId уменьшит на 1 и ответ тоже
    char flag = 0;
    if(calcNeuronId(layersSz, 4, 1) == 2){
        flag = 1;
    }
    else{
        return 0;
    }
    if(calcNeuronId(layersSz, 6, 2) == 0){
        flag = 1;
    }
    else{
        return 0;
    }
    if(calcNeuronId(layersSz, 0, 0) == 0){
        flag = 1;
    }
    else{
        return 0;
    }
    return flag;
    //return calcNeuronId(layersSz, 3, 1);
}

int calculatePreviousLayerStartTest(__global const float* input,
                                    __global int* spikes,
                                    __global const int* layersSz,
                                    __global int* spikesReferrence){
    spikesReferrence = calcPrevLayerStart(input, spikes, layersSz, 2, 2, 1);
    if(spikesReferrence == spikes + 4){
            return 1;
    }
    else{
        return 0;
    }
}

int calculateWeightsStartTest(__global const float* weights,
                              __global int* layersSz,
                              __global const float* referrence) {
    referrence = calcWeightsStart(weights, layersSz, 1, 1, 1);
    int flag = 0;
    if (referrence != weights + 2){
        return 0;
    }
    referrence = calcWeightsStart(weights, layersSz, 1, 2, 3);
    if (referrence != weights + 20){
        return 0;
    }
    return 1;
}

__kernel void tests(__global const int* layersSz,
                    __global const float* weights,
                    __global int* results,
                    __global int* spikes,
                    __global const float* input,
                    __global const float* referrence,
                    __global int* spikesReferrence){
    results[0] = refractoryFunctionTest();
    results[1] = spikeFunctionTest();
    results[2] = calculateSpikesNumberTest(layersSz);
    results[3] = calculateLayerIdTest(layersSz);
    results[4] = calculateNeuronIdTest(layersSz);
    results[5] = calculatePreviousLayerStartTest(input, spikes, layersSz, referrence);
    results[6] = calculateWeightsStartTest(weights, layersSz, referrence);
}