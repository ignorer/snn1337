#pragma once

/* Example

\code
 #include "FullyConnectedNN.h"
 #include "NetworkInitializer.h"

 int main() {
    NetworkInitializer ni("res/network_test");
    FullyConnectedNN network = ni.getNetwork();
 }
\endcode

*/

#include <string>
#include <fstream>
#include <vector>

#include "Network.h"

using namespace std;

class NetworkInitializer {
private:
    string filename;
    Network network;

    int layersNumber;
    vector<int> sizes;
    vector<Layer*> layers;
    int synapsesPerConnection;
    int maxSpikesPerSynapse;
    int exitTime;
    float threshold;

    static const string THRESHOLD_TAG;
    static const string LAYERS_NUMBER_TAG;
    static const string SYNAPSES_PRE_CONNECTION_TAG;
    static const string MAX_SPIKES_PER_CONNECTION_TAG;
    static const string EXIT_TIME_TAG;
    static const string WEIGHTS_TAG;
    static const string INPUT_LAYER_TAG;
    static const string DENSE_LAYER_TAG;
    static const string CONV_LAYER_TAG;
    static const string POOL_LAYER_TAG;
    static const string FILTER_SIZE_TAG;
    static const string NUM_FILTERS_TAG;
    static const string POOL_SIZE_TAG;
    static const string NUM_UNITS_TAG;
    static const string SHAPE_TAG;
    static const string INPUT_SHAPE_TAG;

    void getParams(ifstream& in);
    void getNetworkFromParams();
    void loadFullyConnectedNN(ifstream& in);
    void loadDenseLayer(ifstream& in);
    void loadConvLayer(ifstream& in);
    void loadPoolLayer(ifstream& in);
    void loadInputLayer(ifstream& in);

public:
    NetworkInitializer(string filename);
    const Network &getNetwork() const;
};