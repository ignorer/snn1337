#include <iterator>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "FullyConnectedNN.h"

using namespace std;

FullyConnectedNN::FullyConnectedNN() {
}

FullyConnectedNN::FullyConnectedNN(int layersNumber,
    std::vector<int> sizes,
    std::vector<Layer> layers,
    int synapsesPerConnection,
    int maxSpikesPerSynapse,
    int exitTime,
    float threshold) :
    layersNumber(layersNumber),
    sizes(sizes),
    layers(layers),
    synapsesPerConnection(synapsesPerConnection),
    maxSpikesPerSynapse(maxSpikesPerSynapse),
    exitTime(exitTime),
    threshold(threshold) {
}

int FullyConnectedNN::getLayersNumber() const {
    return layersNumber;
}

const std::vector<int> &FullyConnectedNN::getSizes() const {
    return sizes;
}

const std::vector<Layer> &FullyConnectedNN::getLayers() const {
    return layers;
}

int FullyConnectedNN::getSynapsesPerConnection() const {
    return synapsesPerConnection;
}

int FullyConnectedNN::getMaxSpikesPerSynapse() const {
    return maxSpikesPerSynapse;
}

int FullyConnectedNN::getExitTime() const {
    return exitTime;
}

float FullyConnectedNN::getThreshold() const {
    return threshold;
}

vector<float> FullyConnectedNN::getAllWeights() {
    vector<float> allWeights;
    for (auto& layer : layers) {
        for (auto& neuron : layer.getNeurons()) {
            for (auto& weight : neuron.getWeights()) {
                allWeights.push_back(weight);
            }
        }
    }
    return allWeights;
}

void FullyConnectedNN::print() {
    cerr << "threshold: " << threshold << "\n";
    cerr << "layersNumber: " << layersNumber << "\n";
    cerr << "synapsesPerConnection: " << synapsesPerConnection << "\n";
    cerr << "exitTime: " << exitTime << "\n";
    if (layersNumber != sizes.size()) {
        throw logic_error("layersNumber should be equal to the length of sizes");
    }
    cerr << "sizes: ";
    for (int i = 0; i < layersNumber; i++) {
        cerr << sizes[i] << " ";
    }
    cerr << "\n";
    if (layersNumber != layers.size()) {
        throw logic_error("layersNumber should be equal to the length of layers");
    }
    for (auto& layer : layers) {
        cerr << "weights from layer:\n";
        for (auto& neuron : layer.getNeurons()) {
            cerr << "weights from neuron: ";
            for (auto& weigth : neuron.getWeights()) {
                cerr << weigth << " ";
            }
            cerr << "\n";
        }
    }
    cerr << "allWeights: ";
    vector<float> allWeights = getAllWeights();
    for (int i = 0; i < allWeights.size(); i++) {
        cerr << allWeights[i] << " ";
    }
}
