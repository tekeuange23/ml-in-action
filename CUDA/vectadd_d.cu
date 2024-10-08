#include <iostream>
#include <chrono>
#include <cmath>
#include <cuda.h>

/**
 * #define N 1_000_000_000, #crashed due to insuffisiant memory.
 * PROOF:
 * a @float: 32_bits == 4_bytes
 * # @total:  1_000_000_000 * 3 == 3_000_000_000 variables
 * @conclusion: require 12_000_000_000 bytes of global memory
 *                   ==     11_718_750 KB    of global memory
 *                   ==      11_444.09 MB    of global memory
 *                   ==         11.175 GB    of global memory
*/
#define N 10
// #define N 100000000

using namespace std;


// Host Vector Initialization
float* vectInit(float value, int n) {
    size_t size = n * sizeof(float);
    float* vect = (float*)malloc(size * sizeof(float));

    if (vect == NULL) {
        printf("Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    } else {
        int i;
        for (i = 0; i < n; i++)
            vect[i] = i * value;
    }

    return vect;
}

// Print the array elements
void vectPrint(float* vect, int n) {
    int i;
    printf("-->: ");
    for (i = 0; i < n; i++)
        printf("%f ", vect[i]);
    printf("\n");
}

// Kernel Function:  Each Thread performs one pair wise addition
__global__
void vectAddKernel(float* C, float* A, float* B, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        C[i] = A[i] + B[i];
}

// Compute vector sum C = A+B
void vectAdd(float* C, float* A, float* B, int n, bool cuda) {
    if (!cuda) {
        int i;
        for (i = 0; i < n; i++)
            C[i] = A[i] + B[i];
        vectPrint(A, N);
        vectPrint(B, N);
        vectPrint(C, N);
        return;
    }

    /** ******************************************** */
    /**         Part_1: Allocate Device Memory       */
    /** ******************************************** */
    size_t size = n * sizeof(float);
    float* d_A, * d_B, * d_C;
    /**
     * copy the object from the @host_memory to the @local_memory
     * and save the value of the pointer to the @local_memory  */
    cudaError_t err_A = cudaMalloc((void**)&d_A, size);
    cudaError_t err_B = cudaMalloc((void**)&d_B, size);
    cudaError_t err_C = cudaMalloc((void**)&d_C, size);
    if (err_A != cudaSuccess || err_B != cudaSuccess || err_C != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err_A), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    /** ******************************************** */
    /**           Part_2: Kernel launch code         */
    /** ******************************************** */
    vectAddKernel << <ceil(n / 256.0), 256 >> > (d_C, d_A, d_B, N);

    /** ******************************************** */
    /**            Part_3: Copy in Host Memory       */
    /** ******************************************** */
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    vectPrint(A, N);
    vectPrint(B, N);
    vectPrint(C, N);

    /** ******************************************** */
    /**            Final: Free device vectors        */
    /** ******************************************** */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

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
    // nvcc -ccbin /usr/bin/g++-11 vectadd_d.cu -o vectadd_d && ./vectadd_d
}