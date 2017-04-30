#pragma once

#include <tuple>

#include "Layer.h"

using namespace std;

class InputLayer : public Layer {
  private:
    tuple<int, int, int> shape;
    int size;
  public:
    InputLayer(tuple<int, int, int> shape);

    int getSize();

    tuple<int, int, int> getShape();

    int getFilterSize();

    int getNumFilters();

    int getPoolSize();
};
