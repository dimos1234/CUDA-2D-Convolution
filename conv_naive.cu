#include <iostream>
#include <iomanip>
#include <cstdlib>

// global tells the cpu that this is a kernal function, this will be executed on the gpu
__global__ void naiveConvolution(const unsigned char *d_input, unsigned char *d_output, int width, int height, const float *d_filter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int out_idx = y * width + x;

    // Make sure this thread is actually handling a real pixel
    if (x < width && y < height) {
        float sum = 0.0f;

        for (int f_y = -1; f_y <= 1; f_y++) {
            for (int f_x = -1; f_x <= 1; f_x++) {

                int neighbor_x = x + f_x;
                int neighbor_y = y + f_y;

                if (neighbor_x >= 0 && neighbor_x < width && neighbor_y >= 0 && neighbor_y < height) {

                    int filter_idx = (f_y + 1) * 3 + (f_x + 1);
                    int neighbor_idx = neighbor_y * width + neighbor_x;

                    sum += (float)d_input[neighbor_idx] * d_filter[filter_idx];
                }
                // If it's out of bounds, we do nothing (which treats it as adding 0)
            }
        }

        // Clamp or cast the result back to an unsigned char pixel value
        if (sum > 255.0f) sum = 255.0f;
        if (sum < 0.0f)   sum = 0.0f;
        d_output[out_idx] = (unsigned char)sum;
    }
}

int main() {
  // A tiny small image
  const int width = 512;
  const int height = 512;

  // Total sizes in bytes
  size_t img_size = width * height * sizeof(unsigned char);
  size_t filter_size = 3 * 3 * sizeof(float);

  unsigned char *h_input = (unsigned char*)malloc(img_size);
  unsigned char *h_output = (unsigned char*)malloc(img_size);

  // Initialize the large image array with random mock pixel data
  for (int i = 0; i < width * height; i++) {
      h_input[i] = (unsigned char)(rand() % 256);
  }

  // A simple sharpening/edge filter
  float h_filter[9] = {
        0, -1,  0,
      -1,  5, -1,
        0, -1,  0
  };

  // --- DEVICE (GPU) POINTER DECLARATIONS ---
  unsigned char *d_input, *d_output;
  float *d_filter;

  // --- YOUR CHALLENGE STARTS HERE ---
  // Use cudaMalloc to allocate memory on the GPU for our three pointers.
  // Syntax reminder: cudaMalloc(&pointer, size_in_bytes);

  cudaMalloc(&d_input, img_size);
  cudaMalloc(&d_output, img_size);
  cudaMalloc(&d_filter, filter_size);

  cudaMemcpy(d_input, h_input, img_size, cudaMemcpyHostToDevice);

  cudaMemcpy(d_filter, h_filter, filter_size, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks(32, 32);

  // setup a timer
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // start the timer
  cudaEventRecord(start);

  naiveConvolution<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height, d_filter);

  // stop the timer
  cudaEventRecord(stop);
  cudaEventSynchronize(stop); // Wait for the GPU to completely finish doing math

  // calculate elapsed time
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << "Kernel execution time: " << milliseconds << " ms\n";

  // Clean up stopwatches
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);

/*  std::cout << "Convolution Result Matrix (4x4):\n";
  for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
          std::cout << std::setw(4) << (int)h_output[y * width + x] << " ";
      }
      std::cout << "\n";
  }*/

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_filter);

  free(h_input);
  free(h_output);

  return 0;
}
