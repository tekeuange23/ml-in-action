#ifndef MATRIX_H
#define MATRIX_H

#pragma once
#include <iostream>
#include <cmath>
#include <cuda.h>

#define N 10
#define M 10

using namespace std;

// get card Information
void getCardInfo();

// Host Vector Initialization
float* vectInit(float value, int n);

// Print the array elements
void vectPrint(float* vect, int n);

// Kernel Function:  Each Thread performs one pair wise addition
__global__ void vectAddKernel(float* C, float* A, float* B, int n);

// Compute vector sum C = A+B
void vectAdd(float* C, float* A, float* B, int n, bool cuda);


#endif // MATRIX_H