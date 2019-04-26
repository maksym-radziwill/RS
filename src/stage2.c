#include <arb.h>
#include <stdio.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <sys/sysinfo.h>
#include <curses.h>

#ifdef HAS_MPI
#include <mpi.h>
#endif

#include <unistd.h>
#include <semaphore.h>
#include "aux.h"
#include "status_bar.h"
#include "cuda.h"

void * stage2_thread(void *); 

static int thread_count; 
static fmpz_t global_n;
static fmpz_t maximal_n; 
static pthread_mutex_t n_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t t_mutex = PTHREAD_MUTEX_INITIALIZER; 
static pthread_barrier_t barrier; 
static int world_rank;
static int world_size;

extern void start_kernel(unsigned long blocks, struct computation_data * device_computation_data, struct computation_results * device_results, cudaStream_t stream, int world_rank, int world_size);


/* Given local_n: 
   Update local_n as the new starting point given the current value of global_n
   Update global_n 
   Output in len the length of the new block to be processed 
   Return 1 or 0 depending on whether there is still work to be made or not
*/ 

int determine_increment(fmpz_t len, fmpz_t local_n){
    pthread_mutex_lock(&n_mutex); 
    
    /* Return if the current n value exceeds the maximal one */ 
    if(fmpz_cmp(local_n, maximal_n) >= 0){ 
	pthread_mutex_unlock(&n_mutex);
	return 0; 
    }

    /* Return if the computation has been already exhausted by the other threads */ 
    if(fmpz_cmp(global_n, maximal_n) >= 0){
	pthread_mutex_unlock(&n_mutex);
	return 0; 
    }

    fmpz_set(local_n, global_n); // Set the starting point of current computation

    fmpz_root(len, local_n, 2); // Could take 2 -> 3, 1 -> 32 if we want to be on the safer side
    // set 2 -> 3 then might want to decrease the increment 32 if t is small, say t < 10^21 

    if(fmpz_cmp_ui(len, BLOCK_SIZE) < 0)
	fmpz_set_ui(len, BLOCK_SIZE); 
    
    fmpz_tdiv_q_ui(len, len, BLOCK_SIZE);
    fmpz_mul_ui(len, len, BLOCK_SIZE); 

    fmpz_add(global_n, global_n, len); // Adjust the value of global_n

    /* If it exceeds the maximal one, set it to be maximal and adjust the new length */ 
    if(fmpz_cmp(global_n, maximal_n) >= 0){
	fmpz_set(global_n, maximal_n);
	fmpz_sub(len, global_n, local_n);
       
    }

    pthread_mutex_unlock(&n_mutex);
    return 1; 
}	   

/* Global data shared by all threads */ 

int K;
int totalDevices = 1; 
static arb_t t; 
static double _Complex * result;
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER; 

double _Complex * stage2(arb_t local_t, int k, int world_rnk, int world_sz){

    arb_init(t);
    arb_set(t, local_t); 

    stage2_range(global_n, t); // Set the starting point of the computation  
    
    riemann_siegel_length(maximal_n, t); // Set the finishing point 
    
    result = (double _Complex *) malloc(k*sizeof(double _Complex)); 
    for(int v = 0; v < k; v++) result[v] = 0; 

    if(!nocache_flag)
	load_data(t, k, result, world_rnk, world_sz, 2, global_n); // load data from a file if it exists

    if(fmpz_cmp(global_n, maximal_n) >= 0)
	return result;   
  
    /* Communicate to other threads the world_rank and size */ 

    world_rank = world_rnk;
    world_size = world_sz; 
    
    cudaGetDeviceCount ( &totalDevices );
        
    int num_threads = lcm(get_nprocs(), totalDevices);

    thread_count = num_threads; 

    /* The barrier is only needed if caching is enabled */ 
    
    if(!nocache_flag)
	pthread_barrier_init(&barrier, NULL, num_threads);
    
    K = k;

    pthread_t * threads = (pthread_t *) malloc(num_threads*sizeof(pthread_t)); 
    pthread_t status_bar_thread;  // will not be used if in silent mode
    
    for(int i = 0; i < num_threads; i++){
	int * num = (int *) malloc(sizeof(int));
	*num = i; 
	pthread_create(&threads[i], NULL, &stage2_thread, num); 
    }    

    if(!silent_flag){
	struct status_bar_args args;
	args.maximal_n = &maximal_n;
	args.global_n = &global_n;
	args.n_mutex = &n_mutex; 
	args.world_rank = world_rank;
	args.world_size = world_size; 
	args.str = (char *) malloc(16);
	sprintf(args.str, "Stage 2"); 
	pthread_create(&status_bar_thread, NULL, &status_bar_u, &args); 
    }
	
    for(int i = 0; i < num_threads; i++){
	pthread_join(threads[i], NULL); 
    }

    if(!nocache_flag)
	save_data(t, k, result, world_rank, world_size, 2, global_n); 

    if(!silent_flag)
      	pthread_cancel(status_bar_thread); 
    
    /* Compute the theta function and output the final result */ 
    
    arb_clear(t);
    fmpz_clear(global_n);
    fmpz_clear(maximal_n); 
    
    return result;
}

struct multiplier {
    double sqrt ;
    double _Complex phase; 
};

void compute_multipliers(struct multiplier * multipliers, arb_t * temp, int * aux_prec, fmpz_t n, arb_t local_t){
    
    /* There were some checks here, before */ 
    arb_set_fmpz(temp[3], n); // temp[3] = n;
    arb_floor(temp[3], temp[3], MAX_PREC); // temp[3] = \floor n =: x
    
    /* Fill out the sqrt_multiplier */
    
    do{
	arb_sqrt(temp[4], temp[3], aux_prec[1]); // temp[4] = \sqrt{x}
	arb_inv(temp[4], temp[4], aux_prec[1]); // temp[4] = 1 / \sqrt{x}
    }while(!correct_precision(temp[4], &aux_prec[1]));
    
    multipliers->sqrt = arb_get_d(temp[4]); // sqrt_multiplier = 1 / \sqrt{x}
    
    /* Fill out the oscillating multiplier x^{-it} */
    
    do{
	arb_log(temp[1], temp[3], aux_prec[2]); // temp[1] = \log x
	arb_mul(temp[1], temp[1], local_t, aux_prec[2]); // temp[1] = t \log x
	arb_neg(temp[1], temp[1]); // temp[1] = - t \log x
	arb_cos(temp[2], temp[1], aux_prec[2]); // temp[2] = \cos(-t \log x)
	arb_sin(temp[5], temp[1], aux_prec[2]); // temp[5] = \sin(-t \log x)
    }while((!correct_precision(temp[2], &aux_prec[2]))
	   || !correct_precision(temp[5], &aux_prec[2]));

    multipliers->phase = arb_get_d(temp[2]) + I*arb_get_d(temp[5]); 

}



void compute_taylor_expansions(struct computation_data * host_computation_data, arb_t * temp, int * log_prec, int * sqrt_prec, fmpz_t n, arb_t local_t){

    int taylor_log = LOG_MAX;
    int taylor_sqrt = SQRT_MAX; 
    
    /* Store n in temp[3] */ 
    
    arb_set_fmpz(temp[3], n); // temp[3] = n;
    arb_floor(temp[3], temp[3], MAX_PREC); // temp[3] = \floor n =: x
    
    /* Store -t in temp[0] */
    
    arb_set(temp[0], local_t);
    arb_neg(temp[0], temp[0]);
    
    /* Generate the log_constants[k] */
    
    /* We want to compute
       - t \log ((q (x + h) + a) / (q x))
       = - t \log (1 + h / x + a / q x)
       = - t (log_derivative[0] + log_derivative[1] * (1/x) * h + log_derivative[2] * (1/x)^2 * h^2 + ...
       
       Therefore we will need to store
       - t * log_derivative[0]  \mod 2\pi
       - t * log_derivative[1] / x \mod 2\pi
       - t * log_derivative[2] / x^2 \mod 2\pi
       ...
       
       Because log_derivative[k] = f^{(k)}(1 + a / q x) / k! where f(x) = \log(1 + x) */
    
    /* Recall   temp[0] = -t
       temp[1] = 1/x
       
       In this loop temp[2] and temp[3] will be temporary
       and temp[4] will hold (1/x)^{k}.
       
    */
    
    for(int k = 0; k < taylor_log; k++){
	do{
	    
	    /* Store 1/x in temp[1] */
	    
	    arb_inv(temp[1], temp[3], log_prec[k]); // temp[1] = 1/x
	    arb_pow_ui(temp[4], temp[1], k, log_prec[k]); // temp[4] = (1/x)^k
	    
	    /* Store a / q x in temp[6] */
	    
	    arb_inv(temp[6], temp[3], log_prec[k]); // temp[3] = 1/(q \floor { n / q }) = 1 / (q x)
	    
	    /* Compute 2\pi */
	    
	    arb_const_pi(temp[8], log_prec[k]);
	    arb_mul_ui(temp[8], temp[8], 2, log_prec[k]); // twopi = 2\pi
	    
	    /* Evaluate f^{(k)}(a / q x) / k! */
	    
	    find_log_derivatives(temp[5], temp[6], k, log_prec[k]); // temp[5] has the derivative
	    
	    /* Multiply and mod by 2\pi */
	    
	    arb_mul(temp[2], temp[0], temp[5], log_prec[k]); // temp[2] = -t * log_derivative[k]
	    arb_mul(temp[2], temp[2], temp[4], log_prec[k]); // temp[2] = - t * log_derivative[k] * (1/x)^k
	    arb_fmod(temp[7], temp[2], temp[8], log_prec[k]); // temp[2] = - t * log_derivative[k] * (1/x)^{k} \mod 2\pi
	    
	}while(!correct_precision(temp[7], &log_prec[k]));
	
	host_computation_data->log_constants[k] = arb_get_d(temp[7]); // save temp[2] in table[i].log_constants[k]
	
    }
    
    /* Recall
       temp[1] = 1/x
       temp[6] = a/qx
       
       In this loop temp[2] will be temporary and
       temp[4] will hold (1/x)^{k}.
    */
    
    for(int k = 0; k < taylor_sqrt; k++){
	do{
	    arb_inv(temp[1], temp[3], sqrt_prec[k]); // temp[1] = 1/x
	    arb_pow_ui(temp[4], temp[1], k, sqrt_prec[k]); // temp[4] = (1/x)^k
	    arb_inv(temp[6], temp[3], sqrt_prec[k]); // temp[3] = 1/(\floor { n }) = 1 / x
	    find_sqrt_derivatives(temp[5], temp[6], k, sqrt_prec[k]);
	    arb_mul(temp[2], temp[4], temp[5], sqrt_prec[k]);
	}while(!correct_precision(temp[2], &sqrt_prec[k]));
	
	host_computation_data->sqrt_constants[k] = arb_get_d(temp[2]);
    }
    
}

void * stage2_thread (void * data){ 
 
    pthread_mutex_lock(&mutex); 
    int k = K;
    pthread_mutex_unlock(&mutex); 

    int tid = * ((int *) data);

    cudaSetDevice(tid % totalDevices); 

    int num_var = 10;

    arb_t * temp = init_temp(num_var);
    arb_t * temp_m = init_temp(num_var);
    arb_t * temp_s = init_temp(num_var); 
    
    fmpz_t n, len; fmpz_init(n); fmpz_init(len);  
    arb_t local_t; arb_init(local_t);
    arb_t n_inv; arb_init(n_inv); 
    
    pthread_mutex_lock(&mutex); // Just a precaution
    arb_set(local_t, t);
    pthread_mutex_unlock(&mutex); 
    
    fmpz_set_ui(n, 0);
     
    int log_prec[LOG_MAX]; int sqrt_prec[SQRT_MAX]; 
    
    for(int i = 0; i < LOG_MAX; i++) log_prec[i] = 53;
    for(int i = 0; i < SQRT_MAX; i++) sqrt_prec[i] = 53;

    /* We allocate more than we need to be on the safe side */ 
    
    //    int aux_prec[16]; for(int i = 0; i < 16; i++) aux_prec[i] = 53;
    int aux_prec_m[16]; for(int i = 0; i < 16; i++) aux_prec_m[i] = 53; 
    int aux_prec_s[16]; for(int i = 0; i < 16; i++) aux_prec_s[i] = 53; 
    
    /* Configure CUDA */
    /* Set memory bank to be 64bits natively */ 

    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    
    /* Create a CUDA stream */

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int mem_block = MEM_BLOCK; 

    struct multiplier * multipliers = (struct multiplier *) malloc(mem_block*sizeof(struct multiplier));  
    struct multiplier * shifts = (struct multiplier *) malloc(mem_block*sizeof(struct multiplier)); 
    
    /* Set up pinned memory */

    struct computation_results * host_results;
    struct computation_results * device_results; 
    cudaMallocHost((void **) &host_results, mem_block*sizeof(struct computation_results));
    cudaMalloc((void **) &device_results, mem_block*sizeof(struct computation_results));

    struct computation_data * host_computation_data;
    struct computation_data * device_computation_data; 
    cudaMallocHost((void **) &host_computation_data, sizeof(struct computation_data)*mem_block);
    cudaMalloc((void **) &device_computation_data, sizeof(struct computation_data)*mem_block); 
    
    int l = 0;
    int v = 0;
    static int wait_multiplier = 100; // This is only used if caching is enabled 

    arb_t shift; arb_init(shift);
    arb_set(shift, SHIFT);
    
    while(determine_increment(len, n)){

	host_computation_data[l].len = fmpz_get_ui(len);

	compute_multipliers(&multipliers[l], temp_m, aux_prec_m, n, local_t);  

	compute_multipliers(&shifts[l], temp_s, aux_prec_s, n, shift); 
	
	compute_taylor_expansions(&host_computation_data[l], temp, log_prec, sqrt_prec, n, local_t); 

	l = (l + 1) % mem_block;	
	v = (v + 1) % (wait_multiplier*mem_block);  // The factor 10 decides how frequently we will write to a file
	
	if(l == 0){
	    
	    cudaMemcpyAsync(device_computation_data, host_computation_data, mem_block*sizeof(struct computation_data), cudaMemcpyHostToDevice, stream); 

	    /* Start the cuda kernel */
	    
	    start_kernel(mem_block, device_computation_data, device_results, stream, world_rank, world_size); 
	    
	    /* Retrieve the result of the Cuda computation from (pinned) memory */ 
	    
	    cudaMemcpyAsync(host_results, device_results, mem_block*sizeof(struct computation_results), cudaMemcpyDeviceToHost, stream); 
	    	    
	    cudaStreamSynchronize(stream);
	    
	    /* Lock mutexes and add to the final result */

	    /* would make sense to use avx here */ 
	    
	    pthread_mutex_lock(&mutex);

	    for(int i = 0; i < mem_block; i ++){
		
		double _Complex iterate  = shifts[i].phase;
		double _Complex value =
		    multipliers[i].phase * (host_results[i].cos_results + I*host_results[i].sin_results) * multipliers[i].sqrt;
		
		for(int v = 0; v < k; v++)
		    {
			result[v] += value;
			value *= iterate;
		    }
	    }
	    
	    pthread_mutex_unlock(&mutex);
	    
	}
	
	/* This is only needed if caching is turned on */
	
	if(!nocache_flag)
	    if(v == 0){
		pthread_barrier_wait(&barrier);
		// Do not need to lock mutex because of the barrier
		if(tid == 0) save_data(local_t, k, result, world_rank, world_size, 2, global_n);
		pthread_barrier_wait(&barrier);
	    }
	
    }

    /* A bit inelegant -- exact repetition of the code above with only mem_block = l inserted */
    
    if(l != 0){
	
	mem_block = l;

	cudaMemcpyAsync(device_computation_data, host_computation_data, mem_block*sizeof(struct computation_data), cudaMemcpyHostToDevice, stream); 
	
	/* Start the cuda kernel */
	
	start_kernel(mem_block, device_computation_data, device_results, stream, world_rank, world_size); 
	
	/* Retrieve the result of the Cuda computation from (pinned) memory */ 
	
	cudaMemcpyAsync(host_results, device_results, mem_block*sizeof(struct computation_results), cudaMemcpyDeviceToHost, stream); 	
	cudaStreamSynchronize(stream);
	
	/* Lock mutexes and add to the final result */
	
	pthread_mutex_lock(&mutex);
	for(int i = 0; i < mem_block; i++){
	    double _Complex iterate  = shifts[i].phase;
	    double _Complex value =
		multipliers[i].phase * (host_results[i].cos_results + I*host_results[i].sin_results) * multipliers[i].sqrt; 	   

	    for(int v = 0; v < k; v++){
		result[v] += value;
		value *= iterate;
	    }
	}
	pthread_mutex_unlock(&mutex);    

    }

    /* Clean up the CUDA resources */ 
    
    cudaFreeHost(host_results);
    cudaFree(device_results);

    cudaFreeHost(host_computation_data);
    cudaFree(device_computation_data); 
    
    cudaStreamDestroy(stream);

    clear_temp(temp_m, num_var); 
    clear_temp(temp, num_var);
    fmpz_clear(n);
    fmpz_clear(len);

    /* This is only something that we need to do 
       if caching is turned on, to avoid an infinite wait
       for a barrier which will be never reached */ 
    
    if(!nocache_flag){
	pthread_mutex_lock(&t_mutex); 
	thread_count -= 1;
	pthread_mutex_unlock(&t_mutex); 
	
	while(1){
	    pthread_barrier_wait(&barrier);
	    usleep(1000);
	    pthread_mutex_lock(&t_mutex); 
	    if(thread_count == 0) break;
	    pthread_mutex_unlock(&t_mutex); 
	}
	
	pthread_mutex_unlock(&t_mutex); 
    }
    
    return NULL; 
}
