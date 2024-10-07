#include <stdio.h>
#include <cuda.h>
// #define N 1000000000
#define N 10

// Initialization
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

    /** 
     * copy the object from the @host_memory to the @local_memory 
     * and save the value of the pointer to the @local_memory*/
    cudaError_t err = cudaMalloc((void**)&vect, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    return vect;
}

// Compute vector sum C = A+B
void vecAdd(float* C, float* A, float* B, int n) {
    int i;
    for (i = 0; i < n; i++)
        C[i] = A[i] + B[i];
}

// Print the array elements
void vectPrint(float* vect, int n) {
    int i;
    printf("-->: ");
    for (i = 0; i < n; i++)
        printf("%f ", vect[i]);
    printf("\n");
}

int main() {

    // Init
    float* d_A, * d_B, * d_C;
    d_A = vectInit(1.0, N);
    d_B = vectInit(0.1, N);
    d_C = vectInit(0.0, N);

    /** ******************************************** */
    /**         Part_1: Allocate Device Memory       */
    /** ******************************************** */
    // vecAdd(d_C, d_A, d_B, N);
    /** ******************************************** */

    /** ******************************************** */
    /**           Part_2: Kernel launch code         */
    /** ******************************************** */
    //
    printf("hello\n");
    // vectPrint(d_A, N);
    // vectPrint(d_B, N);
    // vectPrint(d_C, N);
    /** ******************************************** */

    /** ******************************************** */
    /**            Part_3: Free device vectors       */
    /** ******************************************** */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    /** ******************************************** */

    return 0;
    // nvcc -ccbin /usr/bin/g++-11 vectadd_d.cu -o vectadd_d && ./vectadd_d
}