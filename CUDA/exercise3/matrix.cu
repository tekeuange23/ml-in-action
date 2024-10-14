#include <iostream>
#include <cmath>
#include <cuda.h>

#define N 10
#define M 10

using namespace std;

// Print the matrix
void printMatrix(float** mat, int n, int m) {
    printf("==========================================================================================\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++)
            printf("%f ", mat[i][j]);
        printf("\n");
    }
    printf("==========================================================================================\n");
}

// Allocate host_memory for the matrix
float** initMatrix(int n, int m, float val) {
    float** mat;
    mat = (float**)malloc(n * sizeof(float*));
    if (mat == NULL) {
        printf("Error: memory allocation failed in %s at line %d \n", __FILE__, __LINE__);
        exit(1);
    }

    for (int i = 0; i < n; i++) {
        mat[i] = (float*)malloc(m * sizeof(float));
        if (mat[i] == NULL) {
            printf("Error: memory allocation failed in %s at line %d \n", __FILE__, __LINE__);
            exit(1);
        }
        for (int j = 0; j < m; j++) {
            mat[i][j] = val + j;
        }
    }
    return mat;
}

int main() {
    float** A, ** B, ** C;
    A = initMatrix(N, M, 0.0);
    B = initMatrix(N, M, 0.1);
    C = initMatrix(N, M, 1.0);
    printMatrix(A, N, M);
    printMatrix(B, N, M);
    printMatrix(C, N, M);

    return 0;
    // nvcc -ccbin /usr/bin/g++-11 matrix.cu -o matrix_binary && ./matrix_binary
}