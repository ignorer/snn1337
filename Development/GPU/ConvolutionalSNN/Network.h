#pragma once

/* Example

\code
 #include "Network.h"
 #include "Layer.h"

 int main() {
    NetworkInitializer ni("res/network_test");
    Network network = ni.getNetwork();
    vector<float> allWeights = netwotk.getAllWeights();
    vector<int> sizes = netwotk.getAllSizes();
    vector<Layer*> layers = netwotk.getLayers();
 }
\endcode

*/

#include "Layer.h"

class Network {
private:
    int layersNumber;
    std::vector<Layer*> layers;
    int synapsesPerConnection;
    int maxSpikesPerSynapse;
    int exitTime;
    float threshold;

public:
    Network();

    Network(int layersNumber,
         std::vector<Layer*> layers,
         int synapsesPerConnection,
         int maxSpikesPerSynapse,
         int exitTime,
         float threshold);

    void print();

    void printAllWeights();

    int getLayersNumber() const;

    const std::vector<Layer*> &getLayers() const;

    int getSynapsesPerConnection() const;

    int getMaxSpikesPerSynapse() const;

    int getExitTime() const;

    float getThreshold() const;

    std::vector<int> getAllSizes();

    std::vector<float> getAllWeights();

    vector<tuple<int, int, int>> getAllShapes();

    vector<int> getAllNumFilters();

    vector<int> getAllFilterSizes();

    vector<int> getAllPoolSizes();
};

