#include <iostream>
#include <iomanip>
#include <cstdlib>

#define BLOCK_SIZE 16
#define FILTER_RADIUS 1
#define TILE_SIZE 18

__global__ void coalescedConvolution(const unsigned char *d_input, unsigned char *d_output, int width, int height, const float *d_filter) {
    __shared__ unsigned char shared_tile[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    int linear_tid = ty * blockDim.x + tx;

    for (int i = linear_tid; i < TILE_SIZE * TILE_SIZE; i += (BLOCK_SIZE * BLOCK_SIZE)) {
        int tile_row = i / TILE_SIZE;
        int tile_col = i % TILE_SIZE;

        int global_x = (blockIdx.x * blockDim.x - FILTER_RADIUS) + tile_col;
        int global_y = (blockIdx.y * blockDim.y - FILTER_RADIUS) + tile_row;

        if (global_x >= 0 && global_x < width && global_y >= 0 && global_y < height) {
            shared_tile[tile_row][tile_col] = d_input[global_y * width + global_x];
        } else {
            shared_tile[tile_row][tile_col] = 0;
        }
    }

    __syncthreads();

    // add the actual math here

    if (x < width && y < height) {
      float sum = 0.0f;

      #pragma unroll
      for (int f_y = -1; f_y <= 1; f_y++) {
          #pragma unroll
          for (int f_x = -1; f_x <= 1; f_x++) {

              int filter_idx = (f_y + 1) * 3 + (f_x + 1);

              int shared_col = tx + 1 + f_x;
              int shared_row = ty + 1 + f_y;

              sum += (float)shared_tile[shared_row][shared_col] * d_filter[filter_idx];
          }
      }
      if (sum > 255.0f) sum = 255.0f;
      if (sum < 0.0f)   sum = 0.0f;

      int out_idx = y * width + x;
      d_output[out_idx] = (unsigned char)sum;
  }
}

// for python to call
void launch_unrolled_convolution(const unsigned char* input, unsigned char* output, int width, int height, const float* filter) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    // Dynamic ceiling calculation to handle any image size safely
    dim3 numBlocks((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    coalescedConvolution<<<numBlocks, threadsPerBlock>>>(input, output, width, height, filter);

    // Block Python until the GPU is 100% finished doing the math
    cudaDeviceSynchronize();
}

#ifndef BUILDING_PYTHON_MODULE
int main() {
    const int width = 512;
    const int height = 512;
    size_t img_size = width * height * sizeof(unsigned char);
    size_t filter_size = 3 * 3 * sizeof(float);

    unsigned char *h_input = (unsigned char*)malloc(img_size);
    unsigned char *h_output = (unsigned char*)malloc(img_size);

    for (int i = 0; i < width * height; i++) {
        h_input[i] = (unsigned char)(rand() % 256);
    }

    float h_filter[9] = { 0, -1, 0, -1, 5, -1, 0, -1, 0 };

    unsigned char *d_input, *d_output;
    float *d_filter;

    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_output, img_size);
    cudaMalloc(&d_filter, filter_size);

    cudaMemcpy(d_input, h_input, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filter_size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(width / BLOCK_SIZE, height / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    coalescedConvolution<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height, d_filter);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Unrolled Kernel execution time: " << milliseconds << " ms\n";

    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_input); cudaFree(d_output); cudaFree(d_filter);
    free(h_input); free(h_output);

    return 0;
}
#endif
