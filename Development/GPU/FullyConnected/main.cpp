#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <iterator>

#include "FullyConnectedNN.h"
#include "Layer.h"

using namespace std;

FullyConnectedNN loadFullyConnectedNN(const string& filename) {
    ifstream in(filename);

    vector<Layer> layers;

    string strWidth;
    while (getline(in, strWidth)) {
        istringstream inWidth(strWidth);
        int width;
        inWidth >> width;
        string strWeights;
        vector<vector<double>> weights;
        for (int i = 0; i < width; i++) {
            getline(in, strWeights);
            istringstream iss(strWeights);
            vector<double> neuronWeights{istream_iterator<double>{iss}, istream_iterator<double>{}};
            weights.push_back(neuronWeights);
        }
        string strBiases;
        getline(in, strBiases);
        istringstream iss(strBiases);
        vector<double> biases{istream_iterator<double>{iss}, istream_iterator<double>{}};
        layers.push_back(Layer(width, weights, biases));
    }
    return FullyConnectedNN(layers);
}

int main() {
    FullyConnectedNN network = loadFullyConnectedNN("network_xor");
    network.printEmptyValues();
}