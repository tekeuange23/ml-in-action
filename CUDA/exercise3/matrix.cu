#include <iostream>
#include <cmath>
#include <cuda.h>

#define N 10
#define M 10
#define var_name(x) #x

using namespace std;

// Print the matrix
void printMatrix(float** mat, int n, int m, string varName, string event = NULL) {
    printf("%s: %s=\n", event.c_str(), varName.c_str());
    printf("==========================================================================================\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++)
            printf("%f ", mat[i][j]);
        printf("\n");
    }
    printf("==========================================================================================\n");
}

// Allocate matrix in Host_memory
float** initMatrixHost(int n, int m, float val) {
    float** mat;
    mat = (float**)malloc(n * sizeof(float*));
    if (mat == NULL) {
        printf("Error: memory allocation failed in %s at line %d \n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n; i++) {
        mat[i] = (float*)malloc(m * sizeof(float));
        if (mat[i] == NULL) {
            printf("Error: memory allocation failed in %s at line %d \n", __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < m; j++) {
            mat[i][j] = val + j;
        }
    }
    return mat;
}

// Allocate matrix in Host_memory
float** initMatrixDevice(int n, int m) {
    float** d_mat;
    size_t sizeN = n * sizeof(float*);
    size_t sizeM = m * sizeof(float*);

    cudaError_t err = cudaMalloc((void**)&d_mat, sizeN);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n; i++) {
        float* d_matRow;
        cudaError_t err = cudaMalloc((void**)&d_matRow, sizeM);
        if (err != cudaSuccess) {
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
        cudaMemcpy(d_mat[i], d_matRow, sizeM, cudaMemcpyDeviceToDevice);
    }

    return d_mat;
}

void freeMatrixDevice(float** d_mat, int n) {
    for (int i = 0;i > n;i++) {
        cudaFree(d_mat[i]);
    }
    cudaFree(d_mat);
}

// Matrix addition Host A[i][j] = B[i][j] + C[i][j]
void matrixAddHost(float** A, float** B, float** C, int n, int m) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            A[i][j] = B[i][j] + C[i][j];
}

int main() {
    // Matrix Init
    float** A, ** B, ** C;
    A = initMatrixHost(N, M, 0.0);
    B = initMatrixHost(N, M, 0.1);
    C = initMatrixHost(N, M, 1.0);
    printMatrix(B, N, M, var_name(B), "After Initialization");
    printMatrix(C, N, M, var_name(C), "After Initialization");

    // // Matrix Add: HOST
    // matrixAddHost(A, B, C, N, M);
    // printMatrix(A, N, M, var_name(A), "After Host Addition");

    /** ******************************************** */
    /**         Part_1: Allocate Device Memory       */
    /** ******************************************** */
    float** d_B, ** d_C;
    d_B = initMatrixDevice(N, M);
    d_C = initMatrixDevice(N, M);

    return 0;
    // nvcc -ccbin /usr/bin/g++-11 matrix.cu -o matrix_binary && ./matrix_binary
}