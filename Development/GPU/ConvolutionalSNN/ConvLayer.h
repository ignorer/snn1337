#pragma once

#include <vector>

#include "Layer.h"

using namespace std;

class ConvLayer : public Layer {
  private:
    int numFilters;
    int filterSize;
    int size;
    tuple<int, int, int> shape;

  public:
    ConvLayer(Layer* prevLayer, int numFilters, int filterSize);

    int getNumFilters();

    int getFilterSize();

    tuple<int, int, int> getShape();

    int getSize();

    int getPoolSize();
};
