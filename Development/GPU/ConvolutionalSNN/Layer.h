#pragma once

#include <vector>
#include <tuple>

using namespace std;

class Layer {
  protected:
    Layer* prevLayer;
    vector<float> weights;
  public:
    Layer(Layer* prevLayer);

    Layer* getPrevLayer();

    const vector<float> &getWeights();

    void setWeights(vector<float> weights);

    virtual int getSize() = 0;
    virtual tuple<int, int, int> getShape() = 0;
    virtual int getFilterSize() = 0;
    virtual int getNumFilters() = 0;
    virtual int getPoolSize() = 0;
};
