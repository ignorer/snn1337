#pragma once

#include <vector>

#include "Neuron.h"

class Layer {
private:
    int size;
    std::vector<Neuron> neurons;

public:
    Layer(int size, std::vector<Neuron> neurons);

    int getSize() const;
    const std::vector<Neuron> &getNeurons() const;
};
