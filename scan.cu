//ONLY MODIFY THIS FILE!
//YOU CAN MODIFY EVERYTHING IN THIS FILE!

#include "scan.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gpuerrors.h"
#include "gputimer.h"
#include <stdio.h>
#include <stdlib.h>
#include "helper.h"


#define tx threadIdx.x
#define bdx blockDim.x
#define bx blockIdx.x
#define by blockIdx.y
#define gdx gridDim.x

#define Serial 2

uint8_t multiply(uint8_t a, uint8_t b,  uint8_t* alpha_to,  uint8_t* index_of){
	return (a == 0 || b == 0)? 0 :alpha_to[(uint32_t(index_of[a]) + uint32_t(index_of[b])) % 255];
}
__device__ uint8_t deviceMultiply(uint8_t a, uint8_t b,  uint8_t* alpha_to,  uint8_t* index_of){
	return (a == 0 || b == 0)? 0 :alpha_to[(uint32_t(index_of[a]) + uint32_t(index_of[b])) % 255];
}

__device__ void myAdd(uint8_t* A_powers, int power,uint8_t* vector ,uint8_t* ans, uint8_t* alpha_to, uint8_t* index_of){
	uint8_t t;
	uint8_t *matrix = A_powers + (power - 1)*16;
	uint8_t temp[4];
	for (int i=0; i<4; i++){
		t = 0;
		for (int j = 0; j < 4; j++){
			t ^= deviceMultiply(matrix[i * 4 + j], vector[j], alpha_to, index_of);;
		}
		temp[i] = t;	
	}
	for (int i=0; i<4; i++){
		ans[i] ^= temp[i];	
	}
}
void matrix_to_vector_multiply( uint8_t* matrix, uint8_t* v, uint8_t* res, uint8_t* alpha_to, uint8_t* index_of)
{
	uint8_t t;
	for (int i=0; i<4; i++)
	{
		t = 0;
		for (int j = 0; j < 4; j++){
			t ^= multiply(matrix[i*4+j], v[j], alpha_to, index_of);
		}
		res[i] = t;
	}
}
__global__ void kernelFunc(uint8_t* c, uint8_t* A_powers,int step, uint8_t* alpha_to, uint8_t* index_of ){


	__shared__ uint8_t  alpha_to_shared[256];
	__shared__ uint8_t  index_of_shared[256];
	__shared__ uint8_t  A_shared[1024 * 16];
	__shared__ uint8_t  c_shared[1024 * 4];
	
	int id =by * gdx * bdx + bx * bdx + tx;

	for (int i = 0; i < 4; i ++){
		for (int j = 0; j < 4; j++){
			A_shared[tx * 16 + i * 4 + j] = A_powers[tx * 16 + i * 4 + j];
		}
		c_shared[tx * 4 * step + i] = c[id * 4 * step + i];
	}
	
	if (tx < 256)
		alpha_to_shared[tx] = alpha_to[tx]; 
	else if (tx < 512 ){
		index_of_shared[tx - 256] = index_of[tx - 256];
	}
	__syncthreads();
	

	for (int stage = 0; stage < 10; stage++){
		
		int distance = (1 << stage);
		if (distance <= tx){
			myAdd(A_shared,distance,&c_shared[(tx- distance)*4 * step], &c_shared[tx*4* step], alpha_to_shared, index_of_shared);
		}else break;
		
		__syncthreads();
		
	}
	
	for (int i = 0; i < 4; i ++){
		c[id * 4 * step + i] = c_shared[tx * 4 * step+ i] ;
	}
}	
__global__ void makeAdders(uint8_t* c, uint8_t* A_powers,uint8_t* adders ,uint8_t* alpha_to, uint8_t* index_of ){
}
__global__ void addIt(uint8_t* c,uint8_t* adders, uint8_t* A_powers ,uint8_t* alpha_to, uint8_t* index_of ){
	
	__shared__ uint8_t com[4];
	__shared__ uint8_t  A_shared[1024 * 16];
	__shared__ uint8_t  alpha_to_shared[256];
	__shared__ uint8_t  index_of_shared[256];
	
	if (tx == 0){
		for (int i = 0; i < 4; i++){
			com[i] = adders[(bx)* 4 + i];
		}
		
	}
	for (int i = 0; i < 4; i ++){
		for (int j = 0; j < 4; j++){
			A_shared[tx * 16 + i * 4 + j] = A_powers[tx * 16 + i * 4 + j];
		}
	}
	if (tx < 256)
		alpha_to_shared[tx] = alpha_to[tx]; 
	else if (tx < 512 ){
		index_of_shared[tx - 256] = index_of[tx - 256];
	}
	
	__syncthreads();
	
	int id = (bx + 1) * bdx + tx;
	myAdd(A_shared,tx+1,com, &c[id*4], alpha_to_shared, index_of_shared);
	
}


void op(uint8_t* c,int index, const uint8_t* const matrix,uint8_t* alpha_to, uint8_t* index_of){
	for (int i = 0; i < 4; i ++){
		for (int j = 0; j < 4; j ++){
			c[4*index+i] = c[4*index+i] ^ multiply(matrix[i*4+j], c[4*(index - 1)+j],alpha_to,index_of);
		}
	}
}


void addVector(uint8_t* a,uint8_t* b)
{
	for (int i=0; i<4; i++){
		a[i] ^= b[i];	
	}
}
void matrix_to_matrix_multiply(const uint8_t* const a, const uint8_t* const b, uint8_t* c,  uint8_t* alpha_to, uint8_t* index_of)
{
	int dim = 4;
	for (int i=0; i<dim; i++)
	{
		for (int j=0; j<dim; j++)
		{
			c[i*dim+j] = 0;
			for (int k=0; k<dim; k++)
			{
				
			
				c[i*dim+j] ^= multiply(a[i*dim+k], b[k*dim+j], alpha_to,index_of);
			}
		}
	}
}


void copyVector(uint8_t* a,uint8_t*b){
	for (int i = 0; i < 4; i++)
		a[i] = b[i];
}


void gpuKernel(const uint8_t* const a, const uint8_t* const matrix, uint8_t* c, const int m, const int n, uint8_t* alpha_to, uint8_t* index_of)
{
	
	if (m < 20){
		for (int t = 0; t < 4 * n; t ++) 
			c[t] = a[t];
		for (int index = 1; index < n; index ++)			
			op(c,index ,matrix, alpha_to,index_of);
	} 
	else{
		uint8_t  A_powers [1024 * 16];
		for (int i =0; i < 16; i++)
			A_powers[i] = matrix[i];
		for (int i = 0; i < 1024 - 1; i ++)
			matrix_to_matrix_multiply(matrix, &A_powers[(i)*16], &A_powers[(i+1)*16], alpha_to, index_of);
		
		
		uint8_t* c_GPU;
		uint8_t* alpha_to_GPU;
		uint8_t* index_of_GPU;
		uint8_t* A_powers_GPU;
		uint8_t* adders_GPU;

		
		HANDLE_ERROR(cudaMalloc((void**)&c_GPU, 4 * n * sizeof(uint8_t)));
		HANDLE_ERROR(cudaMalloc((void**)&alpha_to_GPU, 256 * sizeof(uint8_t)));
		HANDLE_ERROR(cudaMalloc((void**)&index_of_GPU, 256 * sizeof(uint8_t)));
		HANDLE_ERROR(cudaMalloc((void**)&A_powers_GPU, 4 * 4 * 1024 * sizeof(uint8_t)));
		HANDLE_ERROR(cudaMalloc((void**)&adders_GPU, 64*4 * 1024 * sizeof(uint8_t)));
		

		HANDLE_ERROR(cudaMemcpy(c_GPU, a, 4*n * sizeof(uint8_t), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(alpha_to_GPU, alpha_to, 256 * sizeof(uint8_t), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(index_of_GPU, index_of, 256 * sizeof(uint8_t), cudaMemcpyHostToDevice));		
		HANDLE_ERROR(cudaMemcpy(A_powers_GPU, A_powers, 4*4 *1024 * sizeof(uint8_t), cudaMemcpyHostToDevice));
		
		
		int blockLength = 1024;
		int blockCount = (1<<m - 10); 
		dim3 dimGrid(blockCount/4,4,1); 
		dim3 dimGrid2(blockCount - 1,1,1); 
		dim3 dimBlock(blockLength,1,1);  
		kernelFunc<<< dimGrid,dimBlock >>>(c_GPU, A_powers_GPU,1, alpha_to_GPU, index_of_GPU); 
		
		HANDLE_ERROR(cudaMemcpy(c, c_GPU, 4*n * sizeof(uint8_t), cudaMemcpyDeviceToHost));
		
		
		uint8_t adders[1024*64*4];
		for (int i = 0; i < 16; i++){
			adders[i] = c[1023 * 4 + i];
		}
		for (int i = 1; i < blockCount; i++){
			
			matrix_to_vector_multiply(&A_powers[(1023) * 16], &adders[(i-1)*4], &adders[(i)*4], alpha_to, index_of);
			addVector(&adders[(i)*4],(&c[((i+1) * 1024 - 1) * 4]));			
		}
		
		HANDLE_ERROR(cudaMemcpy(adders_GPU, adders, 1024*64*4  * sizeof(uint8_t), cudaMemcpyHostToDevice));
		
		
		addIt<<< dimGrid2 ,dimBlock >>>(c_GPU,adders_GPU ,A_powers_GPU, alpha_to_GPU, index_of_GPU);
		HANDLE_ERROR(cudaMemcpy(c, c_GPU, 4*n * sizeof(uint8_t), cudaMemcpyDeviceToHost));
		
		
		HANDLE_ERROR(cudaFree(alpha_to_GPU));
		HANDLE_ERROR(cudaFree(index_of_GPU));
		HANDLE_ERROR(cudaFree(A_powers_GPU));
		HANDLE_ERROR(cudaFree(c_GPU));
		HANDLE_ERROR(cudaFree(adders_GPU));
	}
	
}

		