#include <iostream>

#include "NetworkInitializer.h"
#include "Network.h"
#include "Layer.h"
#include "InputLayer.h"
#include "DenseLayer.h"
#include "ConvLayer.h"
#include "PoolLayer.h"

using namespace std;

const string NetworkInitializer::THRESHOLD_TAG = "threshold:";
const string NetworkInitializer::LAYERS_NUMBER_TAG = "layersNumber:";
const string NetworkInitializer::SYNAPSES_PRE_CONNECTION_TAG = "synapsesPerConnection:";
const string NetworkInitializer::MAX_SPIKES_PER_CONNECTION_TAG = "spikesPerSynapse:";
const string NetworkInitializer::EXIT_TIME_TAG = "exitTime:";
const string NetworkInitializer::WEIGHTS_TAG = "weights:";
const string NetworkInitializer::INPUT_LAYER_TAG = "InputLayer:";
const string NetworkInitializer::DENSE_LAYER_TAG = "DenseLayer:";
const string NetworkInitializer::CONV_LAYER_TAG = "ConvLayer:";
const string NetworkInitializer::POOL_LAYER_TAG = "PoolLayer:";
const string NetworkInitializer::FILTER_SIZE_TAG = "filter_size:";
const string NetworkInitializer::NUM_FILTERS_TAG = "num_filters:";
const string NetworkInitializer::NUM_UNITS_TAG = "num_units:";
const string NetworkInitializer::POOL_SIZE_TAG = "pool_size:";
const string NetworkInitializer::INPUT_SHAPE_TAG = "input_shape:";

NetworkInitializer::NetworkInitializer(string filename) {
    cerr << "Initialization begins...\n";
    ifstream in(filename);
    cerr << "File is opening...\n";
    loadFullyConnectedNN(in);
    cerr << "Network is loaded...\n";
}

void NetworkInitializer::loadFullyConnectedNN(ifstream& in) {
    cerr << "Getting params...\n";
    getParams(in);
    cerr << "Creating network from params...\n";
    getNetworkFromParams();
}

void NetworkInitializer::loadInputLayer(ifstream& in) {
    string tag;
    in >> tag;
    int channels, height, width;
    tuple<int, int, int> shape;
    if (tag == NetworkInitializer::INPUT_SHAPE_TAG) {
        in >> channels >> height >> width;
        shape = make_tuple(channels, height, width);
    }
    layers.push_back(new InputLayer(shape));
}

void NetworkInitializer::loadDenseLayer(ifstream& in) {
    string tag;
    in >> tag;
    int size;
    if (tag == NetworkInitializer::NUM_UNITS_TAG) {
        in >> size;
    }
    Layer* prevLayer = layers[layers.size() - 1];
    layers.push_back(new DenseLayer(prevLayer, size));
    in >> tag;
    if (tag == NetworkInitializer::WEIGHTS_TAG) {
        vector<float> weights;
        Layer *denseLayer = layers[layers.size() - 1];
        for (int i = 0; i < denseLayer->getSize(); i++) {
            for (int j = 0; j < prevLayer->getSize(); j++) {
                float weight;
                in >> weight;
                weights.push_back(weight);
            }
        }
        denseLayer->setWeights(weights);
    }
}

void NetworkInitializer::loadConvLayer(ifstream& in) {
    string tag;
    int num_filters, filter_size;
    in >> tag;
    if (tag == NetworkInitializer::FILTER_SIZE_TAG) {
        in >> filter_size;
    }
    in >> tag;
    if (tag == NetworkInitializer::NUM_FILTERS_TAG) {
        in >> num_filters;
    }
    Layer* prevLayer = layers[layers.size() - 1];
    layers.push_back(new ConvLayer(prevLayer, num_filters, filter_size));
    in >> tag;
    if (tag == NetworkInitializer::WEIGHTS_TAG) {
        vector<float> weights;
        Layer *convLayer = layers[layers.size() - 1];
        for (int i = 0; i < convLayer->getNumFilters(); i++) {
            for (int s = 0; s < get<0>(prevLayer->getShape()); s++) {
                for (int j = 0; j < convLayer->getFilterSize(); j++) {
                    for (int z = 0; z < convLayer->getFilterSize(); z++) {
                        float weight;
                        in >> weight;
                        weights.push_back(weight);
                    }
                }
            }
        }
        convLayer->setWeights(weights);
    }
}

void NetworkInitializer::loadPoolLayer(ifstream& in) {
    string tag;
    in >> tag;
    int pool_size;
    if (tag == NetworkInitializer::POOL_SIZE_TAG) {
        in >> pool_size;
    }
    Layer* prevLayer = layers[layers.size() - 1];
    layers.push_back(new PoolLayer(prevLayer, pool_size));
    // weights for PoolLayer are 1 / (pool_size * pool_size)
    vector<float> weights;
    Layer *poolLayer = layers[layers.size() - 1];
    for (int s = 0; s < get<0>(prevLayer->getShape()); s++) {
        for (int j = 0; j < poolLayer->getPoolSize(); j++) {
            for (int z = 0; z < poolLayer->getPoolSize(); z++) {
                float weight;
                weight = 1.0 / (poolLayer->getPoolSize() * poolLayer->getPoolSize()
                    * get<0>(prevLayer->getShape()));
                weights.push_back(weight);
            }
        }
    }
    poolLayer->setWeights(weights);
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
        if (tag == NetworkInitializer::DENSE_LAYER_TAG) {
            cerr << "Loading DenseLayer...\n";
            loadDenseLayer(in);
        }
        if (tag == NetworkInitializer::CONV_LAYER_TAG) {
            cerr << "Loading ConvLayer...\n";
            loadConvLayer(in);
        }
        if (tag == NetworkInitializer::POOL_LAYER_TAG) {
            cerr << "Loading PoolLayer...\n";
            loadPoolLayer(in);
        }
        if (tag == NetworkInitializer::INPUT_LAYER_TAG) {
            cerr << "Loading InputLayer...\n";
            loadInputLayer(in);
        }
    }
}

void NetworkInitializer::getNetworkFromParams() {
    network = Network(layersNumber,
        layers,
        synapsesPerConnection,
        maxSpikesPerSynapse,
        exitTime,
        threshold);
}

const Network &NetworkInitializer::getNetwork() const {
    return network;
}