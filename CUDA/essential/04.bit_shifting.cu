#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 10000000  // Vector size = 10 million
// #define N 5 
#define BLOCK_SIZE 256

// Example:
// A = [1, 2, 3, 4, 5]
// A = [2, 3, 4, 5, 5] <--

// CPU vector addition
void bit_shift_cpu(float *a, float *b, int n) {
    for (int i = 0; i < n-1; i++) {
        b[i] = a[i+1];
    }
}

// CUDA kernel for vector addition
__global__ void bit_shift_gpu(float *a, float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n-1) {
        float temp = a[i+1];
        // __syncthreads;
        b[i] = temp;
        // __syncthreads;
    }
}

// Initialize vector with random values
void init_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
        // vec[i] = (float)i;
    }
}
// Print vector values
void print_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        printf("| %f |", vec[i]);
    }
    printf("\n");
}

// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    float *h_a, *h_b_cpu, *h_b_gpu;
    float *d_a, *d_b;
    size_t size = N * sizeof(float);

    // Allocate host memory
    h_a = (float*)malloc(size);
    h_b_cpu = (float*)malloc(size);
    h_b_gpu = (float*)malloc(size);

    // Initialize vectors
    srand(time(NULL));
    init_vector(h_a, N);

    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);

    // Copy data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        bit_shift_cpu(h_a, h_b_cpu, N);
        bit_shift_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, N);
        cudaDeviceSynchronize();
    }

    // Benchmark CPU implementation
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        bit_shift_cpu(h_a, h_b_cpu, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        bit_shift_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    // Print results
    printf("CPU average time: %f milliseconds\n", cpu_avg_time*1000);
    printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    // Verify results (optional)
    cudaMemcpy(h_b_gpu, d_b, size, cudaMemcpyDeviceToHost);
    // print_vector(h_a, N);
    // print_vector(h_b_cpu, N);
    // print_vector(h_b_gpu, N);
    bool correct = true;
    for (int i = 0; i < N-1; i++) {
        if (fabs(h_b_cpu[i] - h_b_gpu[i]) > 1e-5) { 
            correct = false;
            break;
        }
    }
    printf("Results are %s\n", correct ? "correct" : "incorrect");

    // Free memory
    free(h_a);
    free(h_b_cpu);
    free(h_b_gpu);
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}
