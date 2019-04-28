#include <arb.h>
#include <sys/sysinfo.h>
#include <pthread.h>
#include "config.h"
#ifdef HAS_MPI
#include <mpi.h>
#endif

#include <unistd.h>
#include <sys/time.h>
#include "status_bar.h"
#include <ncurses.h>
#include "aux.h"
#include "cuda.h"

struct thread_arguments {
    double _Complex * result;
    arb_t t;
    unsigned int r;
    unsigned int q;
    int k; 
    int tid; 
};

static int world_rank;
static int world_size; 

double _Complex exp_itlogn(arb_t t, fmpz_t n, arb_t * temp, int * prec){
    arb_set_fmpz(temp[1], n); // temp[1] = n
    
    do{
	arb_log(temp[2], temp[1], *prec); // temp[2] = \log n
	arb_mul(temp[2], temp[2], t, *prec); // temp[2] = t \log n
	arb_mul_si(temp[2], temp[2], -1, *prec); // temp[2] = - t \log n
	arb_sin_cos(temp[3], temp[2], temp[2], *prec); // temp[2] = cos(- t \log n) , temp[3] = sin(- t \log n)
    }while(!correct_precision(temp[2], prec)
	   || !correct_precision(temp[3], prec));
    
    return (arb_get_d(temp[2]) + I * arb_get_d(temp[3]));     
}

fmpz_t * thread_n;
fmpz_t maximal_n; 

void * riemann_siegel_progression_mpfr(void * data){
    struct thread_arguments * th_args = (struct thread_arguments *) data; 

    int tid = th_args->tid; 
    
    arb_t * temp = init_temp(4);

    int prec1 = 53; int prec2 = 53; 
    
    //    fmpz_t n; fmpz_init(n);
    fmpz_set_ui(thread_n[tid], th_args->r); // n = 1 + r

    if(fmpz_cmp_ui(thread_n[tid], 0) == 0) fmpz_add_ui(thread_n[tid],thread_n[tid],th_args->q); 
    
    fmpz_t N; fmpz_init(N);
    stage2_range(N, th_args->t); 
    
    // Set the shift
    
    arb_t delta; arb_init(delta); 
    arb_set(delta, SHIFT); // For now \delta = 0.01 hardcoded 

    double _Complex * result;
    result = (double _Complex *) malloc(th_args->k*sizeof(double _Complex)); 

    for(int v = 0; v < th_args->k; v++)
	result[v] = 0; 
        
    while(fmpz_cmp(thread_n[tid], N) <= 0){
	
	double _Complex value 
	    = exp_itlogn(th_args->t, thread_n[tid], temp, &prec1) / sqrt((double) fmpz_get_ui(thread_n[tid]));

	// I wonder if the division by \sqrt{n} affect precision (i.e how precise is it?)
	double _Complex multiplier 
	    = exp_itlogn(delta, thread_n[tid], temp, &prec2);
	
	for(int v = 0; v < th_args->k; v++){  
	    result[v] += value;
	    value *= multiplier; 
	}
	
	fmpz_add_ui(thread_n[tid],thread_n[tid],th_args->q); // n = n + q;
	
    }

    th_args->result = result; 

    clear_temp(temp, 4);
    arb_clear(delta);
    fmpz_clear(N);
    //    fmpz_clear(thread_n[tid]); 

    return NULL; 

}

double _Complex * stage1(arb_t t, int k, int world_rnk, int world_sz){

    /* Declare world_rank and world_size for other threads */

    world_rank = world_rnk;
    world_size = world_sz; 
    
    int num_threads = get_nprocs(); 

    struct thread_arguments * th_args = (struct thread_arguments *) malloc(num_threads*sizeof(struct thread_arguments));
    pthread_t * threads = (pthread_t *) malloc(num_threads*sizeof(pthread_t)); 
    pthread_t status_bar_thread;
    
    thread_n = (fmpz_t *) malloc(num_threads*sizeof(fmpz_t));
    for(int i = 0; i < num_threads; i++) fmpz_init(thread_n[i]); 
    fmpz_init(maximal_n); stage2_range(maximal_n, t); 

    double _Complex * result;
    result = (double _Complex *) malloc(k*sizeof(double _Complex)); 
    
    for(int v = 0; v < k; v++) result[v] = 0; 

    fmpz_t dummy; fmpz_init(dummy); 

    if(!nocache_flag)
	if(load_data(t, k, result, world_rnk, world_sz, 1, dummy))
	    return result; 

    
    for(int i = 0; i < num_threads; i++){
	pthread_attr_t tattr = thread_priority(99);
	arb_init(th_args[i].t); arb_set(th_args[i].t, t);
	th_args[i].tid = i; 
	th_args[i].r = world_size*i + world_rank;
	th_args[i].q = num_threads*world_size;
	th_args[i].k = k; 
	pthread_create(&threads[i], &tattr, &riemann_siegel_progression_mpfr, &th_args[i]); 
    }

    if(!silent_flag){
	struct status_bar_args args;
	pthread_attr_t tattr = thread_priority(1); 
	args.maximal_n = &maximal_n;
	args.global_n = &thread_n[0];
	args.n_mutex = (pthread_mutex_t *) malloc(sizeof(pthread_mutex_t));
	pthread_mutex_init(args.n_mutex, NULL); 
	args.world_rank = world_rank;
	args.world_size = world_size;
	args.str = (char *) malloc(16);
	sprintf(args.str, "Stage 1"); 
	pthread_create(&status_bar_thread, &tattr, &status_bar_u, &args); 
    }
	
    for(int i = 0; i < num_threads; i++){
    	pthread_join(threads[i], NULL); 
    }

    /* This waits for all the status bar threads to terminate */ 
    
    if(!silent_flag)
      pthread_join(status_bar_thread, NULL); 
    
    for(int i = 0; i < num_threads; i++) fmpz_clear(thread_n[i]); 
    
    for(int i = 0; i < num_threads; i++){
	for(int v = 0; v < k; v++)
	    result[v] += th_args[i].result[v]; 
       	free(th_args[i].result);
    }

    if(!nocache_flag)
	save_data(t, k, result, world_rnk, world_sz, 1, maximal_n);

    free(threads); 
    
    return result; 
    
}
