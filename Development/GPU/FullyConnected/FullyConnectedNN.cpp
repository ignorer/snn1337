#include <iterator>
#include <algorithm>
#include <sstream>
#include <vector>
#include <fstream>
#include <iostream>

#include "FullyConnectedNN.h"
#include "Layer.h"

FullyConnectedNN::FullyConnectedNN(const std::vector<Layer>& layers) : layers(layers){
}

void FullyConnectedNN::printFullyConnectedNN() {
    for (auto& layer : layers) {
        std::cout << layer.getWidth() << "\n";
        for (int i = 0; i < layer.getWidth(); i++) {
            for (int j = 0; j < layer.getWeights()[i].size(); j++) {
                std::cout << layer.getWeights()[i][j] << " ";
            }
            std::cout << "\n";
        }
        for (int i = 0; i < layer.getBiases().size(); i++) {
            std::cout << layer.getBiases()[i] << " ";
        }
        std::cout << "\n";
    }
}

void FullyConnectedNN::printFullyConnectedNNAllWeights() {
    std::vector<double> allWeights = getAllWeights();
    for (auto& weight : allWeights) {
        std::cout << weight << " ";
    }
}

void FullyConnectedNN::printFullyConnectedNNEmptyValues() {
    std::vector<double> emptyValues = getEmptyValues();
    for (auto& value : emptyValues) {
        std::cout << value << " ";
    }
}

std::vector<int> FullyConnectedNN::getSizes() {
    std::vector<int> sizes(layers.size());
    for (auto& layer : layers) {
        sizes.push_back(layer.getWidth());
    }
    return sizes;
}

std::vector<double> FullyConnectedNN::getAllWeights() {
    std::vector<double> allWeights;
    for (auto& layer : layers) {
        int inLayerSize = 0;
        int outLayerSize = layer.getWeights().size();
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

std::vector<double> FullyConnectedNN::getEmptyValues() {
    std::vector<double> values;
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
