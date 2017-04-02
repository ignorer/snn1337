#include "Layer.h"

using namespace std;

Layer::Layer(int width, const vector<vector<double>>& weights, const vector<double>& biases) :
        width(width),
        weights(weights),
        biases(biases) {
}

const vector<vector<double>>& Layer::getWeights() const {
    return weights;
}

const vector<double>& Layer::getBiases() const {
    return biases;
}

int Layer::getWidth() const {
    return width;
}

void Layer::setWeights(const vector<vector<double>>& weights) {
    Layer::weights = weights;
}

void Layer::setBiases(const vector<double>& biases) {
    Layer::biases = biases;
}

void Layer::setWidth(int width) {
    Layer::width = width;
}