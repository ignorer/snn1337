#include "Layer.h"

Layer::Layer(int width, const std::vector<std::vector<double>>& weights,
    const std::vector<double>& biases) : width(width), weights(weights), biases(biases) {
}

const std::vector<std::vector<double>>& Layer::getWeights() const {
    return weights;
}

const std::vector<double>& Layer::getBiases() const {
    return biases;
}

int Layer::getWidth() const {
    return width;
}

void Layer::setWeights(const std::vector<std::vector<double>>& weights) {
    Layer::weights = weights;
}

void Layer::setBiases(const std::vector<double>& biases) {
    Layer::biases = biases;
}

void Layer::setWidth(int width) {
    Layer::width = width;
}