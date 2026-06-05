#include <iostream>
#include <iomanip>
#include <cstdlib>

#define BLOCK_SIZE 16
#define FILTER_RADIUS 1 // 3x3 filter extends 1 pixel out on every side
#define TILE_SIZE (BLOCK_SIZE + 2 * FILTER_RADIUS)

__global__ void tiledConvolution(const unsigned char *d_input, unsigned char *d_output, int width, int height, const float *d_filter) {
  __shared__ unsigned char shared_tile[18][18];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int x = blockIdx.x * blockDim.x + tx;
  int y = blockIdx.y * blockDim.y + ty;

  // assign into the shared tile matrix all the primary assigned pixels for each thread
  if (x < width && y < height) {
    shared_tile[ty + 1][tx + 1] = d_input[y * width + x];
  } else {
    shared_tile[ty+1][tx+1] = 0;
  }

  // left halo border
  if (tx == 0) {
    int halo_x = x - 1; // left row for the left halo
    int halo_y = y;

    if (halo_x >= 0 && halo_y >= 0 && halo_y < height) {
      shared_tile[ty + 1][0] = d_input[halo_y * width + halo_x];
    } else {
      shared_tile[ty + 1][0] = 0;
    }
  }

  // right halo border
  if (tx == BLOCK_SIZE - 1) {
    int halo_x = x + 1; // right row for right halo
    int halo_y = y;

    if (halo_x < width && halo_y >= 0 && halo_y < height) {
      shared_tile[ty + 1][BLOCK_SIZE + 1] = d_input[halo_y * width + halo_x];
    } else {
      shared_tile[ty + 1][BLOCK_SIZE + 1] = 0;
    }
  }

  // top halo border
  if (ty == 0) {
    int halo_x = x;
    int halo_y = y - 1; // Look 1 row up globally

    if (halo_x >= 0 && halo_x < width && halo_y >= 0) {
      shared_tile[0][tx + 1] = d_input[halo_y * width + halo_x];
    } else {
      shared_tile[0][tx + 1] = 0;
    }
  }

  // bottom halo border
  if (ty == BLOCK_SIZE - 1) {
    int halo_x = x;
    int halo_y = y + 1; // Look 1 row down globally

    if (halo_x >= 0 && halo_x < width && halo_y < height) {
      shared_tile[BLOCK_SIZE + 1][tx + 1] = d_input[halo_y * width + halo_x];
    } else {
      shared_tile[BLOCK_SIZE + 1][tx + 1] = 0;
    }
  }

  // 4 corners
  if (tx == 0 && ty == 0) { // Top-Left
      shared_tile[0][0] = (x - 1 >= 0 && y - 1 >= 0) ? d_input[(y - 1) * width + (x - 1)] : 0;
  }
  if (tx == BLOCK_SIZE - 1 && ty == 0) { // Top-Right
      shared_tile[0][BLOCK_SIZE + 1] = (x + 1 < width && y - 1 >= 0) ? d_input[(y - 1) * width + (x + 1)] : 0;
  }
  if (tx == 0 && ty == BLOCK_SIZE - 1) { // Bottom-Left
      shared_tile[BLOCK_SIZE + 1][0] = (x - 1 >= 0 && y + 1 < height) ? d_input[(y + 1) * width + (x - 1)] : 0;
  }
  if (tx == BLOCK_SIZE - 1 && ty == BLOCK_SIZE - 1) { // Bottom-Right
      shared_tile[BLOCK_SIZE + 1][BLOCK_SIZE + 1] = (x + 1 < width && y + 1 < height) ? d_input[(y + 1) * width + (x + 1)] : 0;
  }

  // waits for all other threads IN THE BLOCK to reach this point before continuing
  __syncthreads();

  if (x < width && y < height) {
    float sum = 0.0f;

    for (int f_y = -1; f_y <= 1; f_y++) {
        for (int f_x = -1; f_x <= 1; f_x++) {

            int filter_idx = (f_y + 1) * 3 + (f_x + 1);

            // Hint: Your center is at (ty + 1, tx + 1). You need to offset by f_y and f_x.
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

    float h_filter[9] = {
         0, -1,  0,
        -1,  5, -1,
         0, -1,  0
    };

    unsigned char *d_input, *d_output;
    float *d_filter;

    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_output, img_size);
    cudaMalloc(&d_filter, filter_size);

    cudaMemcpy(d_input, h_input, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filter_size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(width / BLOCK_SIZE, height / BLOCK_SIZE);

    // TIMING OPERATORS
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    tiledConvolution<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height, d_filter);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Tiled Kernel execution time: " << milliseconds << " ms\n";

    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);
    free(h_input);
    free(h_output);

    return 0;
}
