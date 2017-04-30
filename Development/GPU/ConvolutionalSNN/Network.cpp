#include <iterator>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "Network.h"

using namespace std;

Network::Network() {
}

Network::Network(int layersNumber,
       std::vector<Layer*> layers,
       int synapsesPerConnection = 1,
       int maxSpikesPerSynapse = 5,
       int exitTime = 500,
       float threshold = 0.5) :
    layersNumber(layersNumber),
    layers(layers),
    synapsesPerConnection(synapsesPerConnection),
    maxSpikesPerSynapse(maxSpikesPerSynapse),
    exitTime(exitTime),
    threshold(threshold) {
}

int Network::getLayersNumber() const {
    return layersNumber;
}

const std::vector<Layer*> &Network::getLayers() const {
    return layers;
}

int Network::getSynapsesPerConnection() const {
    return synapsesPerConnection;
}

int Network::getMaxSpikesPerSynapse() const {
    return maxSpikesPerSynapse;
}

int Network::getExitTime() const {
    return exitTime;
}

float Network::getThreshold() const {
    return threshold;
}

std::vector<int> Network::getAllSizes() {
    vector<int> allSizes;
    for (auto& layer : layers) {
        allSizes.push_back(layer->getSize());
    }
    return allSizes;
}

vector<float> Network::getAllWeights() {
    vector<float> allWeights;
    for (auto &layer : layers) {
        for (auto &weight : layer->getWeights()) {
            allWeights.push_back(weight);
        }
    }
    return allWeights;
}

vector<tuple<int, int, int>> Network::getAllShapes() {
    vector<tuple<int, int, int>> allShapes;
    for (auto& layer : layers) {
        allShapes.push_back(layer->getShape());
    }
    return allShapes;
}

vector<int> Network::getAllNumFilters() {
    vector<int> allNumFilters;
    for (auto& layer : layers) {
        allNumFilters.push_back(layer->getNumFilters());
    }
    return allNumFilters;
}

vector<int> Network::getAllFilterSizes() {
    vector<int> allFilterSizes;
    for (auto& layer : layers) {
        allFilterSizes.push_back(layer->getFilterSize());
    }
    return allFilterSizes;
}

vector<int> Network::getAllPoolSizes() {
    vector<int> allPoolSizes;
    for (auto& layer : layers) {
        allPoolSizes.push_back(layer->getPoolSize());
    }
    return allPoolSizes;
}

void Network::print() {
    cerr << "threshold: " << threshold << "\n";
    cerr << "layersNumber: " << layersNumber << "\n";
    cerr << "synapsesPerConnection: " << synapsesPerConnection << "\n";
    cerr << "exitTime: " << exitTime << "\n";
    if (layersNumber != getAllSizes().size()) {
        throw logic_error("layersNumber should be equal to the length of sizes");
    }
    cerr << "\nallSizes: \n";
    for (int i = 0; i < layersNumber; i++) {
        cerr << layers[i]->getSize() << " ";
    }
    cerr << "\n";
    if (layersNumber != layers.size()) {
        throw logic_error("layersNumber should be equal to the length of layers");
    }
    for (auto &layer : layers) {
        cerr << "weights from layer:\n";
        for (auto &weigth : layer->getWeights()) {
            cerr << weigth << " ";
        }
        cerr << "\n";
    }
//    cerr << "allWeights: \n";
//    vector<float> allWeights = getAllWeights();
//    for (int i = 0; i < allWeights.size(); i++) {
//        cerr << allWeights[i] << " ";
//    }
    cerr << "\nallNumFilters: \n";
    vector<int> allNumFilters = getAllNumFilters();
    for (int i = 0; i < allNumFilters.size(); i++) {
        cerr << allNumFilters[i] << " ";
    }
    cerr << "\nallFilterSizes: \n";
    vector<int> allFilterSizes = getAllFilterSizes();
    for (int i = 0; i < allFilterSizes.size(); i++) {
        cerr << allFilterSizes[i] << " ";
    }
    cerr << "\nallPoolSizes: \n";
    vector<int> allPoolSizes = getAllPoolSizes();
    for (int i = 0; i < allPoolSizes.size(); i++) {
        cerr << allPoolSizes[i] << " ";
    }
    cerr << "\nallShapes: \n";
    vector<tuple<int, int, int>> allShapes = getAllShapes();
    for (int i = 0; i < allShapes.size(); i++) {
        cerr << get<0>(allShapes[i]) << " " << get<1>(allShapes[i]) << " " << get<2>(allShapes[i]) << "\n";
    }
}