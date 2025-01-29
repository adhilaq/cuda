#include <iostream>
#include <cuda_runtime.h>

__global__ void rgb2Gray_kernel(unsigned char *red, unsigned char *green, unsigned char *blue, unsigned char *gray, unsigned int width, unsigned int height)
{
    unsigned int row = blockDim.y*blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x*blockIdx.x + threadIdx.x;
    
    if(col < width && row < height)
    {
        unsigned int i = row*width + col;
        gray[i] = red[i]*0.21f + green[i]*0.72f + blue[i]*0.07f;
    }
}

void rgb2Gray_GPU(unsigned char *red, unsigned char *green, unsigned char *blue, unsigned char *gray, unsigned int width, unsigned int height)
{
    //Allocate GPU Memory
    unsigned char *red_d, *green_d, *blue_d, *gray_d;
    cudaMalloc((void**)&red_d, sizeof(unsigned char)*height*width);
    cudaMalloc((void**)&green_d, sizeof(unsigned char)*height*width);
    cudaMalloc((void**)&blue_d, sizeof(unsigned char)*height*width);
    cudaMalloc((void**)&gray_d, sizeof(unsigned char)*height*width);

    //Copy to GPU
    cudaMemcpy(red_d, red, sizeof(unsigned char)*height*width, cudaMemcpyHostToDevice);
    cudaMemcpy(green_d, green, sizeof(unsigned char)*height*width, cudaMemcpyHostToDevice);
    cudaMemcpy(blue_d, blue, sizeof(unsigned char)*height*width, cudaMemcpyHostToDevice);

    //Call the GPU kernel
    dim3 numThreadsPerBlock(32,32);
    dim3 numBlocks((width + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x, (height + numThreadsPerBlock.y - 1)/numThreadsPerBlock.y);
    rgb2Gray_kernel<<< numBlocks, numThreadsPerBlock >>>(red_d, green_d, blue_d, gray_d, width, height);

    //Copy from the GPU
    cudaMemcpy(gray, gray_d, sizeof(unsigned char)*height*width, cudaMemcpyDeviceToHost);

    //Deallocate GPU Memory
    cudaFree(red_d);
    cudaFree(green_d);
    cudaFree(blue_d);
    cudaFree(gray_d);
}


int main(int argc, char **argv) 
{
    unsigned int width = (argc > 1)?(atoi(argv[1])):(1<<5);
    unsigned int height = (argc > 1)?(atoi(argv[1])):(1<<5);

    unsigned char* red = (unsigned char*) malloc(width*height*sizeof(unsigned char));
    unsigned char* green = (unsigned char*) malloc(width*height*sizeof(unsigned char));
    unsigned char* blue = (unsigned char*) malloc(width*height*sizeof(unsigned char));
    unsigned char* gray = (unsigned char*) malloc(width*height*sizeof(unsigned char));

    for(int i=0; i<height; ++i)
        for(int j=0; j<width;++j)
        {
            //Stored in row major order
            red[width*j + i] = (unsigned char) rand();
            green[width*j + i] =  (unsigned char)rand();
            blue[width*j + i] = (unsigned char) rand();
        }

    rgb2Gray_GPU(red, green, blue, gray, width, height);

    // Cleanup
    free(red);
    free(green);
    free(blue);
    free(gray);
    
    return 0;
}
