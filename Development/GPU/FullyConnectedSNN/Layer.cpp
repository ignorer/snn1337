#include "Layer.h"

using namespace std;

Layer::Layer(int size, std::vector<Neuron> neurons) :
        size(size),
        neurons(neurons) {
}

int Layer::getSize() const {
    return size;
}

const std::vector<Neuron> &Layer::getNeurons() const {
    return neurons;
}
