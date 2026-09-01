
// ============================================================
// Step 2: CUDA Matrix Multiplication Program
// ============================================================

#include <iostream>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication.
// Each GPU thread computes one element of matrix C.
__global__ void matrixMulKernel(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    // Calculate the row handled by this thread.
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate the column handled by this thread.
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure the thread is inside the matrix dimensions.
    if (row < N && col < N) {

        float sum = 0.0f;

        // Compute one element of the output matrix.
        // C[row, col] = sum of A[row, k] * B[k, col]
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }

        // Store the calculated value in matrix C.
        C[row * N + col] = sum;
    }
}

// ============================================================
// CPU Matrix Multiplication
// ============================================================

// This function performs matrix multiplication on the CPU.
// It is used as the baseline for comparing GPU performance.

void matrixMulCPU(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    // Initialize output matrix C to zero.
    for (size_t i = 0; i < (size_t)N * N; i++) {
        C[i] = 0.0f;
    }

    // Cache-friendly loop order: row -> k -> column.
    for (int row = 0; row < N; row++) {

        for (int k = 0; k < N; k++) {

            // Read this value from A once and reuse it.
            float a_value = A[row * N + k];

            for (int col = 0; col < N; col++) {

                // Accumulate the matrix multiplication result.
                C[row * N + col] +=
                    a_value * B[k * N + col];
            }
        }
    }
}

// ============================================================
// Main Function
// Allocates matrices and initializes input values
// ============================================================

int main(int argc, char** argv) {

    // Default matrix size.
    int N = 256;

    // If a matrix size is provided from the command line,
    // use that value instead.
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    // Total number of elements in one N x N matrix.
    size_t numElements = (size_t)N * N;

    // Total memory required for one matrix in bytes.
    size_t bytes = numElements * sizeof(float);

    // Allocate CPU memory for input matrices A and B,
    // CPU result matrix, and GPU result matrix.
    float* h_A = new float[numElements];
    float* h_B = new float[numElements];
    float* h_C_cpu = new float[numElements];
    float* h_C_gpu = new float[numElements];

    // Initialize input matrices with simple values.
    // Using fixed values makes the computation reproducible.
    for (size_t i = 0; i < numElements; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    // Print the matrix size being tested.
    std::cout << "Matrix size: "
              << N << " x " << N << std::endl;

// ============================================================
// Step 5: Allocate GPU Memory and Copy Inputs to GPU
// ============================================================

// Device pointers for matrices A, B, and C.
float* d_A;
float* d_B;
float* d_C;

// Allocate memory on the GPU for all three matrices.
cudaMalloc((void**)&d_A, bytes);
cudaMalloc((void**)&d_B, bytes);
cudaMalloc((void**)&d_C, bytes);

// Copy input matrix A from CPU memory to GPU memory.
cudaMemcpy(
    d_A,
    h_A,
    bytes,
    cudaMemcpyHostToDevice
);

// Copy input matrix B from CPU memory to GPU memory.
cudaMemcpy(
    d_B,
    h_B,
    bytes,
    cudaMemcpyHostToDevice
);

// ============================================================
// Step 6: Configure CUDA Blocks and Threads
// and Launch the GPU Kernel
// ============================================================

// Each CUDA block contains 16 x 16 = 256 threads.
// Each thread computes one element of output matrix C.
dim3 threadsPerBlock(16, 16);

// Calculate how many blocks are required to cover
// the complete N x N output matrix.
dim3 blocksPerGrid(
    (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
    (N + threadsPerBlock.y - 1) / threadsPerBlock.y
);

// Launch the CUDA matrix multiplication kernel.
//
// <<<blocksPerGrid, threadsPerBlock>>>
// tells CUDA how many blocks and threads to use.
matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(
    d_A,
    d_B,
    d_C,
    N
);

// Wait until all GPU threads have finished.
// This is important before reading the GPU result.
cudaDeviceSynchronize();

// Copy the completed result matrix C
// from GPU memory back to CPU memory.
cudaMemcpy(
    h_C_gpu,
    d_C,
    bytes,
    cudaMemcpyDeviceToHost
);

// ============================================================
// Step 7: Run CPU Version and Check Correctness
// ============================================================

// Run the same matrix multiplication on the CPU.
matrixMulCPU(
    h_A,
    h_B,
    h_C_cpu,
    N
);

// Compare the CPU and GPU results.
// If both implementations are correct, corresponding
// elements should be approximately equal.
bool correct = true;

for (size_t i = 0; i < numElements; i++) {

    // Small tolerance is used because floating-point
    // calculations can have tiny numerical differences.
    if (fabs(h_C_cpu[i] - h_C_gpu[i]) > 1e-3) {
        correct = false;
        break;
    }
}

// Print whether the GPU computation matches the CPU result.
if (correct) {
    std::cout << "Result check: PASSED" << std::endl;
} else {
    std::cout << "Result check: FAILED" << std::endl;
}

// ============================================================
// Step 8: Timing Measurements
// ============================================================

// ---------- CPU timing ----------

// CUDA events are not used for CPU timing.
// We use the C++ high-resolution clock instead.
auto cpu_start = std::chrono::high_resolution_clock::now();

matrixMulCPU(
    h_A,
    h_B,
    h_C_cpu,
    N
);

auto cpu_end = std::chrono::high_resolution_clock::now();

// Convert CPU execution time to milliseconds.
double cpu_ms =
    std::chrono::duration<double, std::milli>(
        cpu_end - cpu_start
    ).count();


// ---------- GPU timing ----------

// Create CUDA events for accurate GPU timing.
cudaEvent_t start, stop;

cudaEventCreate(&start);
cudaEventCreate(&stop);


// Measure Host-to-Device transfer time.
cudaEventRecord(start);

cudaMemcpy(
    d_A,
    h_A,
    bytes,
    cudaMemcpyHostToDevice
);

cudaMemcpy(
    d_B,
    h_B,
    bytes,
    cudaMemcpyHostToDevice
);

cudaEventRecord(stop);
cudaEventSynchronize(stop);

float h2d_ms = 0.0f;
cudaEventElapsedTime(&h2d_ms, start, stop);


// Measure only the GPU kernel execution time.
cudaEventRecord(start);

matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(
    d_A,
    d_B,
    d_C,
    N
);

cudaEventRecord(stop);
cudaEventSynchronize(stop);

float kernel_ms = 0.0f;
cudaEventElapsedTime(&kernel_ms, start, stop);


// Measure Device-to-Host transfer time.
cudaEventRecord(start);

cudaMemcpy(
    h_C_gpu,
    d_C,
    bytes,
    cudaMemcpyDeviceToHost
);

cudaEventRecord(stop);
cudaEventSynchronize(stop);

float d2h_ms = 0.0f;
cudaEventElapsedTime(&d2h_ms, start, stop);


// Total GPU transfer time.
float transfer_ms = h2d_ms + d2h_ms;

// End-to-end GPU time includes kernel + transfers.
float gpu_total_ms = kernel_ms + transfer_ms;

// Calculate end-to-end speedup.
double speedup = cpu_ms / gpu_total_ms;


// Print timing results.
std::cout << "CPU time (ms): " << cpu_ms << std::endl;
std::cout << "GPU kernel time (ms): " << kernel_ms << std::endl;
std::cout << "H2D + D2H time (ms): " << transfer_ms << std::endl;
std::cout << "GPU end-to-end time (ms): " << gpu_total_ms << std::endl;
std::cout << "End-to-end speedup: " << speedup << "x" << std::endl;

// ============================================================
// Step 10: Clean Up Memory and Finish Program
// ============================================================

// Destroy CUDA timing events.
cudaEventDestroy(start);
cudaEventDestroy(stop);

// Free memory allocated on the GPU.
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);

// Free memory allocated on the CPU.
delete[] h_A;
delete[] h_B;
delete[] h_C_cpu;
delete[] h_C_gpu;

// Reset the CUDA device and release remaining GPU resources.
cudaDeviceReset();

// Return 0 means the program completed successfully.
return 0;

}  // End of main()
