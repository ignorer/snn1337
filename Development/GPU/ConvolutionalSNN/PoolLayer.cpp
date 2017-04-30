#include "PoolLayer.h"
#include "Layer.h"

PoolLayer::PoolLayer(Layer* prevLayer, int poolSize) :
    Layer(prevLayer),
    poolSize(poolSize) {
    shape = make_tuple(get<0>(prevLayer->getShape()),
        get<1>(prevLayer->getShape()) / poolSize,
        get<2>(prevLayer->getShape()) / poolSize);
    size = get<0>(shape) * get<1>(shape) * get<2>(shape);
}

int PoolLayer::getPoolSize() {
    return poolSize;
}

int PoolLayer::getSize() {
    return size;
}

int PoolLayer::getFilterSize() {
    return -1;
}

int PoolLayer::getNumFilters() {
    return -1;
}

tuple<int, int, int> PoolLayer::getShape() {
    return shape;
}