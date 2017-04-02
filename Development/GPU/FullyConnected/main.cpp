#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>
#include <iterator>

#include <FullyConnectedNN.h>

FullyConnectedNN loadFullyConnectedNN(const std::string& filename) {
    std::ifstream in(filename);

    std::vector<Layer> layers;

    std::string strWidth;
    while (getline(in, strWidth)) {
        std::istringstream inWidth(strWidth);
        int width;
        inWidth >> width;
        std::string strWeights;
        std::vector<std::vector<double>> weights;
        for (int i = 0; i < width; i++) {
            getline(in, strWeights);
            std::istringstream iss(strWeights);
            std::vector<double> neuronWeights{std::istream_iterator<double>{iss},
                                  std::istream_iterator<double>{}};
            weights.push_back(neuronWeights);
        }
        std::string strBiases;
        getline(in, strBiases);
        std::istringstream iss(strBiases);
        std::vector<double> biases{std::istream_iterator<double>{iss},
                                    std::istream_iterator<double>{}};
        layers.push_back(Layer(width, weights, biases));
    }
    return FullyConnectedNN(layers);
}

int main() {
    FullyConnectedNN network = loadFullyConnectedNN("network_xor");
    network.printFullyConnectedNNEmptyValues();
}