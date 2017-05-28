#include "Neuron.h"

Neuron::Neuron(const std::vector<float>& weights) : weights(weights) {
}

const std::vector<float> &Neuron::getWeights() const {
    return weights;
}