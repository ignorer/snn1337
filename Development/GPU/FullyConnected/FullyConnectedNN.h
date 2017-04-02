#pragma once

#include <vector>

class Layer;

class FullyConnectedNN {
private:
    std::vector<Layer> layers;
public:
    FullyConnectedNN(const std::vector<Layer>& layers);

    void print();

    void printAllWeights();

    void printEmptyValues();

    std::vector<int> getSizes();

    std::vector<double> getAllWeights();

    std::vector<double> getEmptyValues();
};
