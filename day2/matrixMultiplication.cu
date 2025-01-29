#include <iostream>
#include <cuda_runtime.h>

__global__ void matmul_kernel(float *A, float *B, float *C, unsigned int N)
{
    int outRow = blockDim.y*blockIdx.y + threadIdx.y;
    int outCol = blockDim.x*blockIdx.x + threadIdx.x;
        
    if(outCol < N && outRow < N)
    {
        float sum = 0.0f;
        for(int i=0; i<N; ++i)
        {
          if(inRow < N && inCol < N)
          {
              sum += A[outRow*N + i]*B[i*N + outCol];
          }
        }

        C[outRow*N + outCol] = sum;
    }
}

void matmul_GPU(float *A, float *B, float *C, unsigned int N)
{
    //Allocate GPU Memory
    float *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, sizeof(float)*N*N);
    cudaMalloc((void**)&B_d, sizeof(float)*N*N);
    cudaMalloc((void**)&C_d, sizeof(float)*N*N);

    //Copy to GPU
    cudaMemcpy(A_d, A, sizeof(float)*N*N, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, sizeof(float)*N*N, cudaMemcpyHostToDevice);

    //Call the GPU kernel
    dim3 numThreadsPerBlock(32,32);
    dim3 numBlocks((N + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x, (N + numThreadsPerBlock.y - 1)/numThreadsPerBlock.y);
    matmul_kernel<<< numBlocks, numThreadsPerBlock >>>(A_d, B_d, C_d, N);

    //Copy from the GPU
    cudaMemcpy(C, C_d, sizeof(float)*N*N, cudaMemcpyDeviceToHost);

    //Deallocate GPU Memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}


int main() 
{
    unsigned int N = 32;

    float *A = (float*) malloc(N*N*sizeof(float));
    float *B = (float*) malloc(N*N*sizeof(float));
    float *C = (float*) malloc(N*N*sizeof(float));

    for(int i=0; i<N; ++i)
        for(int j=0; j<N;++j)
        {
            //Stored in row major order
            A[N*j + i] = (float) rand();
            B[N*j + i] = (float) rand();
        }

    matmul_GPU(A, B, C, N);

    // Cleanup
    free(A);
    free(B);
    free(C);
    
    return 0;
}

