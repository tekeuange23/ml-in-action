#include "matrix.h"
#include <chrono>

int main() {
    // get card info
    // getCardInfo();

    // Init
    float* A, * B, * C;
    A = vectInit(1.0, N);
    B = vectInit(0.1, N);
    C = vectInit(0.0, N);

    // Addition on Host
    auto start_time = chrono::high_resolution_clock::now();
    vectAdd(C, A, B, N, false);
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> host_time = end_time - start_time;
    std::cout << "Time on Host:: " << host_time.count() << " seconds." << endl;

    // Addition on Device
    start_time = chrono::high_resolution_clock::now();
    vectAdd(C, A, B, N, true);
    end_time = chrono::high_resolution_clock::now();
    chrono::duration<double>  device_time = end_time - start_time;
    std::cout << "Time on Device:: " << device_time.count() << " seconds." << endl;
    std::cout << endl << "Conclusion: computation on the Device is "
        << host_time.count() / device_time.count()
        << " Faster than the Host." << endl;

    // Free Host Memory
    free(A);
    free(B);
    free(C);

    return 0;
    // nvcc -ccbin /usr/bin/g++-11 vectadd_d.cu -o vectadd_d_binary && ./vectadd_d_binary
}