#pragma once

#include <vector>

class Layer {
private:
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    int width;
public:
    Layer(int width, const std::vector<std::vector<double>>& weights, const std::vector<double>& biases);

    const std::vector<std::vector<double>>& getWeights() const;

    const std::vector<double>& getBiases() const;

    int getWidth() const;

    void setWeights(const std::vector <std::vector<double>>& weights);

    void setBiases(const std::vector<double>& biases);

    void setWidth(int width);
};
