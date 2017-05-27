#pragma once

/* Example

\code
 #include "InputReader.h"

 int main() {
    InputReader ir;
    ir.read();
    vector<vector<int>> trainImagesData = ir.getTestImagesData();
    vector<int> trainImagesLabels = ir.getTestImagesLabels();
    //every input vector should be transformed to the vector of frequencies
    //frequencies are integers from 125 to 250
    vector<int> freq = ir.getFrequencies(trainImagesData[0]);
 }
\endcode

*/

#include <mnist_reader.hpp>

using namespace std;

class InputReader {
  private:
    vector<vector<int>> testImagesData;
    vector<int> testImagesLabels;
  public:
    void read();

    static float castToFloat(int c);
    static int castToInt(unsigned char c);
    static vector<int> castVectorToVectorOfInt(vector<unsigned char> c);
    static vector<int> getFrequencies(vector<int> pixels);

    const vector<vector<int>> &getTestImagesData();
    vector<vector<float>> getTestImageFloatData();
    const vector<int> &getTestImagesLabels();
};