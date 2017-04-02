#ifndef GPU_FULLYCONNECTEDNN_H
#define GPU_FULLYCONNECTEDNN_H

#include "Layer.h"

class FullyConnectedNN {
private:
    std::vector<Layer> layers;
public:
    FullyConnectedNN(const std::vector<Layer>& layers);

    void printFullyConnectedNN();

    void printFullyConnectedNNAllWeights();

    void printFullyConnectedNNEmptyValues();

    std::vector<int> getSizes();

    std::vector<double> getAllWeights();

    std::vector<double> getEmptyValues();
};

#endif //GPU_FULLYCONNECTEDNN_H
