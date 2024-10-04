#include <stdio.h>
#include <stdlib.h>
// #define N 1000000000
#define N 10

// Initialization
float* vectInit(float value, int n) {
    float* vect = (float*)malloc(n * sizeof(float));

    if (vect == NULL) {
        printf("Memory allocation failed!\n");
        exit(1);
    } else {
        int i;
        for (i = 0; i < n; i++) {
            vect[i] = i * value;
        }
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
    float* h_A, * h_B, * h_C;
    h_A = vectInit(1.0, N);
    h_B = vectInit(0.1, N);
    h_C = vectInit(0.0, N);

    /** *************************** */
    /**         Vect Addition       */
    /** *************************** */
    vecAdd(h_C, h_A, h_B, N); // */
    /** *************************** */

    // Show
    vectPrint(h_A, N);
    vectPrint(h_B, N);
    vectPrint(h_C, N);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}