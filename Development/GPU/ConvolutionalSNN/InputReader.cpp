#include <algorithm>

#include "InputReader.h"

int InputReader::castToInt(unsigned char c) {
    return int(c);
}

vector<int> InputReader::castVectorToVectorOfInt(vector<unsigned char> c) {
    vector<int> transformed(c.size());
    transform(c.begin(), c.end(), transformed.begin(), castToInt);
    return transformed;
}

vector<int> InputReader::getFrequencies(vector<int> pixels) {
    vector<int> transformed(pixels.size());
    for (int i = 0; i < pixels.size(); i++) {
        double trans = pixels[i];
        trans = trans * (125.0 / 255.0);
        trans = trans + 125.0;
        transformed[i] = int(trans);
    }
    return transformed;
}

void InputReader::read() {
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    testImagesData.resize(dataset.test_images.size());
    testImagesLabels.resize(dataset.test_labels.size());

    transform(dataset.test_images.begin(), dataset.test_images.end(),
        testImagesData.begin(), castVectorToVectorOfInt);
    transform(dataset.test_labels.begin(), dataset.test_labels.end(),
        testImagesLabels.begin(), castToInt);
}

const vector<vector<int>> &InputReader::getTestImagesData() {
    return testImagesData;
}

const vector<int> &InputReader::getTestImagesLabels() {
    return testImagesLabels;
}