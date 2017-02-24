#include <iostream>

// very useful macro
#define __CL_ENABLE_EXCEPTIONS

int main() {
    for (int i = 0; i < 10; ++i) {
        std::cout << i << " ";
    }
}