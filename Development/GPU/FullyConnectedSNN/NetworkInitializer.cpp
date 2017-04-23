#include <iostream>

#include "NetworkInitializer.h"
#include "FullyConnectedNN.h"

const string NetworkInitializer::THRESHOLD_TAG = "threshold:";
const string NetworkInitializer::LAYERS_NUMBER_TAG = "layersNumber:";
const string NetworkInitializer::SIZES_TAG = "sizes:";
const string NetworkInitializer::SYNAPSES_PRE_CONNECTION_TAG = "synapsesPerConnection:";
const string NetworkInitializer::MAX_SPIKES_PER_CONNECTION_TAG = "spikesPerSynapse:";
const string NetworkInitializer::EXIT_TIME_TAG = "exitTime:";
const string NetworkInitializer::WEIGHTS_TAG = "weights:";

NetworkInitializer::NetworkInitializer(string filename) {
    std::cerr << "Initialization begins...\n";
    ifstream in(filename);
    std::cerr << "File is opening...\n";
    loadFullyConnectedNN(in);
    std::cerr << "Network is loaded...\n";
}

void NetworkInitializer::loadFullyConnectedNN(ifstream& in) {
    std::cerr << "Getting params...\n";
    getParams(in);
    std::cerr << "Creating network from params...\n";
    getNetworkFromParams();
}

void NetworkInitializer::getParams(ifstream& in) {
    string tag;
    while (in >> tag) {
        if (tag == NetworkInitializer::LAYERS_NUMBER_TAG) {
            in >> layersNumber;
        }
        if (tag == NetworkInitializer::SYNAPSES_PRE_CONNECTION_TAG) {
            in >> synapsesPerConnection;
        }
        if (tag == NetworkInitializer::MAX_SPIKES_PER_CONNECTION_TAG) {
            in >> maxSpikesPerSynapse;
        }
        if (tag == NetworkInitializer::EXIT_TIME_TAG) {
            in >> exitTime;
        }
        if (tag == NetworkInitializer::THRESHOLD_TAG) {
            in >> threshold;
        }
        if (tag == NetworkInitializer::SIZES_TAG) {
            int size;
            for (int i = 0; i < layersNumber; i++) {
                in >> size;
                sizes.push_back(size);
            }
        }
        if (tag == NetworkInitializer::WEIGHTS_TAG) {
            for (int i = 0; i < layersNumber; i++) {
                std::vector<std::vector<float>> weightsLayer;
                if (i != 0) {
                    for (int neuron = 0; neuron < sizes[i]; neuron++) {
                        std::vector<float> weightsNeuron;
                        for (int prev_neuron = 0; prev_neuron < sizes[i - 1]; prev_neuron++) {
                            float weight;
                            in >> weight;
                            weightsNeuron.push_back(weight);
                        }
                        weightsLayer.push_back(weightsNeuron);
                    }
                    weights.push_back(weightsLayer);
                }
            }
        }
    }
}

void NetworkInitializer::getNetworkFromParams() {
    std::vector<Layer> layers;
    for (int i = 0; i < layersNumber; i++) {
        std::vector<Neuron> neurons;
        for (int neuron = 0; neuron < sizes[i]; neuron++) {
            std::vector<float> neuronWeights;
            if (i == 0) {
                neurons.push_back(Neuron(neuronWeights));
            } else {
                neurons.push_back(Neuron(weights[i - 1][neuron]));
            }
        }
        layers.push_back(Layer(neurons.size(), neurons));
    }
    network = FullyConnectedNN(layersNumber,
        sizes,
        layers,
        synapsesPerConnection,
        maxSpikesPerSynapse,
        exitTime,
        threshold);
}

const FullyConnectedNN &NetworkInitializer::getNetwork() const {
    return network;
}
