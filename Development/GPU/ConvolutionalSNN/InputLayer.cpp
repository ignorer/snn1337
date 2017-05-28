#include <tuple>

#include "InputLayer.h"

using namespace std;

InputLayer::InputLayer(tuple<int, int, int> shape) :
    Layer(nullptr),
    shape(shape) {
    size = get<0>(shape) * get<1>(shape) * get<2>(shape);
}

tuple<int, int, int> InputLayer::getShape() {
    return shape;
}

int InputLayer::getNumFilters() {
    return -1;
}

int InputLayer::getFilterSize() {
    return -1;
}

int InputLayer::getPoolSize() {
    return -1;
}

int InputLayer::getSize() {
    return size;
}
