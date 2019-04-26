#include <cuda_runtime.h>
#include "cuda.h"

__inline__ __device__ double warpReduceSum(double val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2)

#if CUDA_VERSION >= 9
      val += __shfl_down_sync(0xffffffff, val, offset);
#else
      val += __shfl_down(val, offset);
#endif

  return val;
}

__device__ double blockReduceSum(double val) {

  static __shared__ double shared[BLOCK_SIZE/32]; // Shared mem for BLOCK_SIZE/32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

  return val;
  }

/* Change this code later so that size is bigger
   i.e address the <<< 1, 32 >>> issue */ 

__global__ void riemann_siegel(struct computation_data * device_computation_data,
	   				          struct computation_results * device_results,
						  int world_rank, int world_size){

    double aux_sum_cos;
    double aux_sum_sin;

    int w_size = world_size;
    int w_rank = world_rank; 

    __shared__ struct computation_data data;

    unsigned long size = BLOCK_SIZE;
    unsigned long b = blockIdx.x; 
    unsigned long l = threadIdx.x;

    aux_sum_cos = 0;
    aux_sum_sin = 0; 

    // copy global memory into shared memory using coallescing
    if(l < (ALIGN_BLOCK / 4))
        *(((int *) &data) + l) = (int) *(((int *) &device_computation_data[b]) + l);

    __syncthreads();

    for(int h = w_rank*size + l; h < data.len; h += w_size*size){
    
	double phase_sin = 0;
	double phase_cos = 0;
	double sqrt_term = 0;
	
	/* Obtain exp(- it log(1 + h / x + a / qx)
	   in phase_sin and phase_cos */
	
	double phase = 0;
	double power = 1;
	for(int k = 0; k < LOG_MAX; k++){ 
	    phase += data.log_constants[k]*power;
	    power = h*power;
	}

	sincos(phase, &phase_sin, &phase_cos); 

	/* Obtain 1/sqrt(1 + h / x + a / qx) in sqrt_part */
	
	power = 1;
	sqrt_term = 0;
	for(int k = 0; k < SQRT_MAX; k++){
	    sqrt_term += data.sqrt_constants[k]*power;
	    power = h*power;
	}

	aux_sum_cos += phase_cos * sqrt_term;
	aux_sum_sin += phase_sin * sqrt_term;
    }

     aux_sum_cos = blockReduceSum(aux_sum_cos);    
     __syncthreads();
     aux_sum_sin = blockReduceSum(aux_sum_sin);

     __syncthreads();

    if(l == 0){
        device_results[b].cos_results = aux_sum_cos;
    	device_results[b].sin_results = aux_sum_sin;
    }
}


extern "C" void start_kernel(unsigned long blocks, struct computation_data * device_computation_data, struct computation_results * device_results, cudaStream_t stream, int world_rank, int world_size){

     	riemann_siegel <<< blocks, BLOCK_SIZE, 0, stream >>> (device_computation_data, device_results, world_rank, world_size); 

}
