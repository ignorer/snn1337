__kernel void neuron(
        __global const int* layersSizes, 
        const int layersNumber,  
        int synapsesPerConnection,
        int spikesPerSynapse, 
        int exitTime,
        __global const float* weights,
        __global const int* spikes,
        float threshold,
        int* t,
        int* sem,
        __global const float* input,
        __global float* output
        ) {
}