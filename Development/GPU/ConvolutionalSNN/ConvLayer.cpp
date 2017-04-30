#include "ConvLayer.h"

ConvLayer::ConvLayer(Layer* prevLayer, int numFilters, int filterSize) :
    Layer(prevLayer),
    numFilters(numFilters),
    filterSize(filterSize) {
    shape = make_tuple(numFilters,
        get<1>(prevLayer->getShape()) - (filterSize - 1), get<2>(prevLayer->getShape()) - (filterSize - 1));
    size = get<0>(shape) * get<1>(shape) * get<2>(shape);
    weights.resize(numFilters * filterSize * filterSize);
}

int ConvLayer::getNumFilters() {
    return numFilters;
}

int ConvLayer::getFilterSize() {
    return filterSize;
}

tuple<int, int, int> ConvLayer::getShape() {
    return shape;
}

int ConvLayer::getPoolSize() {
    return -1;
}

int ConvLayer::getSize() {
    return size;
}