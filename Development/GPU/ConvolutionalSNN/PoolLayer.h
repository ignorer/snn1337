#pragma once

#include <tuple>

#include "Layer.h"

using namespace std;

class PoolLayer : public Layer {
  private:
    int poolSize;
    tuple<int, int, int> shape;
    int size;
  public:
    PoolLayer(Layer* prevLayer, int poolSize);

    int getPoolSize();

    int getSize();

    tuple<int, int, int> getShape();

    int getFilterSize();

    int getNumFilters();
};
