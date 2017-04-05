#include <iterator>
#include <vector>
#include <iostream>

#include "FullyConnectedNN.h"
#include "Layer.h"

using namespace std;

FullyConnectedNN::FullyConnectedNN(const vector<Layer>& layers) : layers(layers){
}

void FullyConnectedNN::print() {
    for (auto& layer : layers) {
        cout << layer.getWidth() << "\n";
        for (int i = 0; i < layer.getWidth(); i++) {
            for (int j = 0; j < layer.getWeights()[i].size(); j++) {
                cout << layer.getWeights()[i][j] << " ";
            }
            cout << "\n";
        }
        for (int i = 0; i < layer.getBiases().size(); i++) {
            cout << layer.getBiases()[i] << " ";
        }
        cout << "\n";
    }
}

void FullyConnectedNN::printAllWeights() {
    vector<float> allWeights = getAllWeights();
    for (auto& weight : allWeights) {
        cout << weight << " ";
    }
}

void FullyConnectedNN::printEmptyValues() {
    vector<float> emptyValues = getEmptyValues();
    for (auto& value : emptyValues) {
        cout << value << " ";
    }
}

vector<size_t> FullyConnectedNN::getSizes() {
    vector<size_t> sizes(layers.size());
    for (auto& layer : layers) {
        sizes.push_back(layer.getWidth());
    }
    return sizes;
}

vector<float> FullyConnectedNN::getAllWeights() {
    vector<float> allWeights;
    for (auto& layer : layers) {
        size_t inLayerSize = 0;
        size_t outLayerSize = layer.getWeights().size();
        if (outLayerSize != 0) {
           inLayerSize = layer.getWeights()[0].size();
        }
        for (int i = 0; i < inLayerSize; i++) {
            allWeights.push_back(layer.getBiases()[i]);
            for (int j = 0; j < outLayerSize; j++) {
                allWeights.push_back(layer.getWeights()[j][i]);
            }
        }
    }
    return allWeights;
}

vector<float> FullyConnectedNN::getEmptyValues() {
    vector<float> values;
    for (int i = 0; i < layers.size(); i++) {
        if (i != layers.size() - 1) {
            values.push_back(1);
        }
        for (int j = 0; j < layers[i].getWidth(); j++) {
            values.push_back(0);
        }
    }
    return values;
}
