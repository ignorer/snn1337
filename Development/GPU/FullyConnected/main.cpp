#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>

class Layer{
private:
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    int width;
public:
    Layer(int width, std::vector<std::vector<double>> weights, std::vector<double> biases) {
        this->width = width;
        for(int i = 0; i < this->width; i++) {
            this->weights = weights;
            this->biases = biases;
        }
    }

    const std::vector<std::vector<double>> &getWeights() const {
        return weights;
    }

    const std::vector<double> &getBiases() const {
        return biases;
    }

    int getWidth() const {
        return width;
    }


    void setWeights(const std::vector<std::vector<double>> &weights) {
        Layer::weights = weights;
    }

    void setBiases(const std::vector<double> &biases) {
        Layer::biases = biases;
    }

    void setWidth(int width) {
        Layer::width = width;
    }
};

class FullyConnectedNN{
private:
    std::vector<Layer> layers;
public:
    FullyConnectedNN(std::vector<Layer> layers) {
        this->layers = layers;
    }

    void printFullyConnectedNN() {
        for (auto layer : layers) {
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
};

FullyConnectedNN loadFullyConnectedNN(std::string filename) {
    std::ifstream in(filename);

    std::vector<Layer> layers;

    std::string strWidth;
    while (getline(in, strWidth)) {
        std::istringstream inWidth(strWidth);
        int width;
        inWidth >> width;
        std::string strWeights;
        std::vector<std::vector<double>> weights;
        for (int i = 0; i < width; i++) {
            getline(in, strWeights);
            std::istringstream iss(strWeights);
            std::vector<double> neuronWeights{std::istream_iterator<double>{iss},
                                  std::istream_iterator<double>{}};
            weights.push_back(neuronWeights);
        }
        std::string strBiases;
        getline(in, strBiases);
        std::istringstream iss(strBiases);
        std::vector<double> biases{std::istream_iterator<double>{iss},
                                    std::istream_iterator<double>{}};
        layers.push_back(Layer(width, weights, biases));
    }
    return FullyConnectedNN(layers);
}

int main() {
    FullyConnectedNN network = loadFullyConnectedNN("network_digits");
}