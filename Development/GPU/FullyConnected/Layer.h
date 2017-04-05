#pragma once

#include <vector>

class Layer {
private:
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;
    int width;

public:
    Layer(int width, const std::vector<std::vector<float>>& weights, const std::vector<float>& biases);

    const std::vector<std::vector<float>>& getWeights() const;

    const std::vector<float>& getBiases() const;

    int getWidth() const;
};
