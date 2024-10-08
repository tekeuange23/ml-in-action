#include "matrix.h"

// get card Information
void getCardInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        exit(EXIT_FAILURE);
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    printf("Device %d: %s\n", 0, deviceProp.name);
    printf("Max threads Dim X: %d\n", deviceProp.maxThreadsDim[0]);
    printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("MultiProcessor count: %d\n", deviceProp.multiProcessorCount);
    printf("Max threads per MultiProcessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
}

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
