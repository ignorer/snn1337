#include <tuple>

#include "DenseLayer.h"

using namespace std;

DenseLayer::DenseLayer(Layer *prevLayer, int size) :
    Layer(prevLayer),
    size(size) {
    weights.resize(size * prevLayer->getSize());
}

int DenseLayer::getSize() {
    return size;
}

int DenseLayer::getFilterSize() {
    return -1;
}

int DenseLayer::getNumFilters() {
    return -1;
}

int DenseLayer::getPoolSize() {
    return -1;
}

tuple<int, int, int> DenseLayer::getShape() {
    return make_tuple<int, int, int>(-1, -1, -1);
}