#include "Layer.h"

using namespace std;

Layer::Layer(Layer* prevLayer) :
    prevLayer(prevLayer) {
}

Layer* Layer::getPrevLayer() {
    return prevLayer;
}

const vector<float> &Layer::getWeights() {
    return weights;
}

void Layer::setWeights(vector<float> weights) {
    this->weights = weights;
}
