#pragma once

#include <vector>

class Neuron {
  private:
    std::vector<float> weights;
  public:
    Neuron(const std::vector<float>& weights);

    const std::vector<float> &getWeights() const;
};