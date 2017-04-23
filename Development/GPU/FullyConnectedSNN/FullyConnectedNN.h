#pragma once

/* Example

\code
 #include "FullyConnectedNN.h"
 #include "NetworkInitializer.h"

 int main() {
    NetworkInitializer ni("res/network_test");
    FullyConnectedNN network = ni.getNetwork();
    network.print(); // print network in stdout to visualize
    network.getAllWeights(); // weights associated with one neuron, which value should be calculated, are close
 }
\endcode

*/

#include "Layer.h"

class FullyConnectedNN {
private:
    int layersNumber;
    std::vector<int> sizes;
    std::vector<Layer> layers;
    int synapsesPerConnection;
    int maxSpikesPerSynapse;
    int exitTime;
    float threshold;

public:
    FullyConnectedNN();

    FullyConnectedNN(int layersNumber,
        std::vector<int> sizes,
        std::vector<Layer> layers,
        int synapsesPerConnection,
        int maxSpikesPerSynapse,
        int exitTime,
        float threshold);

    void print();

    void printAllWeights();

    int getLayersNumber() const;

    const std::vector<int> &getSizes() const;

    const std::vector<Layer> &getLayers() const;

    int getSynapsesPerConnection() const;

    int getMaxSpikesPerSynapse() const;

    int getExitTime() const;

    float getThreshold() const;

    std::vector<float> getAllWeights();
};
