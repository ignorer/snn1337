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

#include "FullyConnectedNN.h"

using namespace std;

class NetworkInitializer {
  private:
    std::string filename;
    FullyConnectedNN network;

    int layersNumber;
    std::vector<int> sizes;
    std::vector<std::vector<std::vector<float>>> weights;
    int synapsesPerConnection;
    int maxSpikesPerSynapse;
    int exitTime;
    float threshold;

    static const string THRESHOLD_TAG;
    static const string LAYERS_NUMBER_TAG;
    static const string SIZES_TAG;
    static const string SYNAPSES_PRE_CONNECTION_TAG;
    static const string MAX_SPIKES_PER_CONNECTION_TAG;
    static const string EXIT_TIME_TAG;
    static const string WEIGHTS_TAG;

    void getParams(ifstream& in);
    void getNetworkFromParams();
    void loadFullyConnectedNN(ifstream& in);

  public:
    NetworkInitializer(string filename);
    const FullyConnectedNN &getNetwork() const;
};
