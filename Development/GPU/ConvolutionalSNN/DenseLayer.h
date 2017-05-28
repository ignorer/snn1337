#pragma once

#include "Layer.h"

class DenseLayer : public Layer {
  private:
    int size;
  public:
    DenseLayer(Layer *prevLayer, int size);

    int getSize();

    tuple<int, int, int> getShape();

    int getFilterSize();

    int getNumFilters();

    int getPoolSize();
};