#include <iostream>
#include <cuda_runtime.h>

__global__ void imageBlur_kernel(unsigned char *image, unsigned char *blur, unsigned int width, unsigned int height)
{
    int BLUR_SIZE = 1;
    int outRow = blockDim.y*blockIdx.y + threadIdx.y;
    int outCol = blockDim.x*blockIdx.x + threadIdx.x;
        
    if(outCol < width && outRow < height)
    {
        unsigned int average = 0;
        for(int inRow=outRow-BLUR_SIZE; inRow<=outRow+BLUR_SIZE; ++inRow)
            for(int inCol=outCol-BLUR_SIZE; inCol<=outCol+BLUR_SIZE; ++inCol)
            {
                if(inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                {
                    average += image[inRow*width + inCol];
                }
            }
        blur[outRow*width + outCol] = (unsigned char) average/((2*BLUR_SIZE+1)*(2*BLUR_SIZE+1));
    }
}

void imageBlur_GPU(unsigned char *image, unsigned char *blur, unsigned int width, unsigned int height)
{
    //Allocate GPU Memory
    unsigned char *image_d, *blur_d;
    cudaMalloc((void**)&image_d, sizeof(unsigned char)*height*width);
    cudaMalloc((void**)&blur_d, sizeof(unsigned char)*height*width);

    //Copy to GPU
    cudaMemcpy(image_d, image, sizeof(unsigned char)*height*width, cudaMemcpyHostToDevice);

    //Call the GPU kernel
    dim3 numThreadsPerBlock(32,32);
    dim3 numBlocks((width + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x, (height + numThreadsPerBlock.y - 1)/numThreadsPerBlock.y);
    imageBlur_kernel<<< numBlocks, numThreadsPerBlock >>>(image_d, blur_d, width, height);

    //Copy from the GPU
    cudaMemcpy(blur, blur_d, sizeof(unsigned char)*height*width, cudaMemcpyDeviceToHost);

    //Deallocate GPU Memory
    cudaFree(image_d);
    cudaFree(blur_d);
}


int main(int argc, char **argv) 
{
    unsigned int width = (argc > 1)?(atoi(argv[1])):(1<<5);
    unsigned int height = (argc > 1)?(atoi(argv[1])):(1<<5);

    unsigned char* image = (unsigned char*) malloc(width*height*sizeof(unsigned char));
    unsigned char* blur = (unsigned char*) malloc(width*height*sizeof(unsigned char));

    for(int i=0; i<height; ++i)
        for(int j=0; j<width;++j)
        {
            //Stored in row major order
            image[width*j + i] = (unsigned char) rand();
        }

    imageBlur_GPU(image, blur, width, height);

    // Cleanup
    free(image);
    free(blur);
    
    return 0;
}
