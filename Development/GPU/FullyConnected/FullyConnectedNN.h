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

    std::vector<size_t> getSizes();

    std::vector<float> getAllWeights();

    std::vector<float> getEmptyValues();

    std::vector<std::vector<float>> getInput(std::string filename);

    std::vector<std::vector<float>> getOutput(std::string filename);
};
