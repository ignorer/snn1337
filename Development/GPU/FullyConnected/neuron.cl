
inline float dot_prod(float* a, float* b, size_t size) {
    float res = 0;
    for (size_t i = 0; i < size; i++) {
        res += a[i]*b[i];
    }
    return res;
}

inline float local_global_dot_prod(float* a, __global float* b, size_t size) {
    float res = 0;
    for (size_t i = 0; i < size; i++) {
        res += a[i]*b[i];
    }
    return res;
}

inline float sigmoid(float x) {
    return 1.7159*tanh(0.66666667*x);
}

__private float* slice(global float* a, size_t offset, size_t size) {
    __private float s[64];
    for (size_t i = offset; i < offset + size; i++) {
        s[i-offset] = a[i];
    }
    return s;
}

size_t array_subsum(global size_t* a, size_t id) {
    size_t res = 0;
    for(size_t i=0; i < id; i++) {
        res += a[i];
    }
    return res;
}

size_t get_neuron_gid(global size_t* layers_sizes, size_t layer_id, size_t neuron_id) {
    return array_subsum(layers_sizes, layer_id) + neuron_id;
}

__kernel void neuron(
    __global const int* layers_sizes,
    __global const float* weights,
    __global int* counters,
    __global float* values,
    __global const float* inputs,
    __global float* outputs) {

    int neuron_id = get_local_id(0);
    int layer_id = get_group_id(0);
    int nb_biases = layer_id + 1;
    __private float* cur_weights = slice(weights, get_neuron_gid(layers_sizes, layer_id, neuron_id) + nb_biases, layers_sizes[layer_id-1]);
    if (layer_id == 0) {
        values[neuron_id + nb_biases] = sigmoid(local_global_dot_prod(cur_weights, inputs, layers_sizes[layer_id]));
    }
    __private float* presynaptic_values = slice(values, get_neuron_gid(layers_sizes, layer_id-1, 0), layers_sizes[layer_id-1]);  // same for layer
    if (layer_id == get_num_groups(0)-1) {
        outputs[neuron_id] =  sigmoid(dot_prod(cur_weights, presynaptic_values, layers_sizes[layer_id]));
    }
    values[get_neuron_gid(layers_sizes, layer_id, neuron_id) + nb_biases] = sigmoid(dot_prod(cur_weights, presynaptic_values, layers_sizes[layer_id]));
    barrier(CLK_LOCAL_MEM_FENCE);
}