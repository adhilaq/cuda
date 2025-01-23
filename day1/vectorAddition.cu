#include <iostream>
#include <cuda_runtime.h>

__global__ void vecAdd_kernel(float *x, float *y, float *z, float N)
{
    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i < N)
    {
        z[i] = x[i] + y[i];
    }
}

void vecAdd_GPU(float *x, float *y, float *z, float N)
{
    //Allocate GPU Memory
    float *x_d, *y_d, *z_d;
    cudaMalloc((void**)&x_d, sizeof(float)*N);
    cudaMalloc((void**)&y_d, sizeof(float)*N);
    cudaMalloc((void**)&z_d, sizeof(float)*N);

    //Copy to GPU
    cudaMemcpy(x_d, x, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, sizeof(float)*N, cudaMemcpyHostToDevice);

    //Run the GPU code
    const unsigned int numThreadsPerBlock = 512;
    const unsigned int numBlocks = (N + numThreadsPerBlock - 1)/numThreadsPerBlock;
    vecAdd_kernel<<< numBlocks, numThreadsPerBlock >>>(x_d, y_d, z_d, N);

    //Copy from the GPU
    cudaMemcpy(z, z_d, sizeof(float)*N, cudaMemcpyDeviceToHost);

    //Deallocate GPU Memory
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
}


int main(int argc, char **argv) 
{
    unsigned int N = (argc > 1)?(atoi(argv[1])):(1<<15);

    float* x = (float*) malloc(N*sizeof(float));
    float* y = (float*) malloc(N*sizeof(float));
    float* z = (float*) malloc(N*sizeof(float));

    for(int i=0; i<N; ++i)
    {
        x[i] = rand();
        y[i] = rand();
    }

    vecAdd_GPU(x, y, z, N);

    // Cleanup
    free(x);
    free(y);
    free(z);
    
    return 0;
}
