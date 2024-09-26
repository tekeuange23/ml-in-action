#include <stdio.h>
#include <stdlib.h>
#define N 10

// Compute vector sum C = A+B
void vecAdd(float *C, float *A, float *B, int n)
{
    int i;
    for (i = 0; i < n; i++)
        C[i] = A[i] + B[i];
}

// Initialization
float* vectInit(float value, size_t size){
    float *vect = (float *)malloc(size * sizeof(float));

    if (vect == NULL)
    {
        printf("Memory allocation failed!\n");
        exit(1);
    }
    else
    {
        int i;
        for (i = 0; i < size; i++)
        {
            vect[i] = i * value; 
        }
    }

    return vect;
}

// Print the array elements
void vectPrint(float *vect)
{
    int i;
    printf("-->: ");
    for (i = 0; i < 10; i++)
    {
        printf("%f ", vect[i]);
    }
    printf("\n");
}

int main()
{

    // Init
    float *A, *B, *C;
    A = vectInit(1.0, N);
    B = vectInit(0.1, N);
    C = vectInit(0.0, N);

    // Vect ddition
    vecAdd(C, A, B, N);

    // Show
    vectPrint(A);
    vectPrint(B);
    vectPrint(C);

    // Free memory
    free(A);
    free(B);
    free(C);

    return 0;
}