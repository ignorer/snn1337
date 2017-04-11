#include "Layer.h"

using namespace std;

Layer::Layer(int width, const vector<vector<float>>& weights, const vector<float>& biases) :
        width(width),
        weights(weights),
        biases(biases) {
}

const vector<vector<float>>& Layer::getWeights() const {
    return weights;
}

const vector<float>& Layer::getBiases() const {
    return biases;
}

int Layer::getWidth() const {
    return width;
}
