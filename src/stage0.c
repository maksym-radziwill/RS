#include "aux.h"
#include <pthread.h>
#include <sys/sysinfo.h>
#include <acb.h>
#include <acb_dirichlet.h>

struct thread_arguments {
    arb_t t;
    int k;
    int tid;
    double _Complex * zeta;
};

void * simple_zeta_thread(void * data){
    struct thread_arguments * th_args = (struct thread_arguments *) data; 

    double _Complex * zeta = th_args->zeta; 
    
    acb_t res; acb_init(res);
    acb_t s; acb_init(s); 
    arb_t onehalf; arb_init(onehalf);
    arb_t realpart; arb_init(realpart);
    arb_t imagpart; arb_init(imagpart); 
    arb_t local_t; arb_init(local_t); 
    acb_t shifti; acb_init(shifti);
    arb_t shift; arb_init(shift); 
    acb_t shift_temp; acb_init(shift_temp); 
    
    int num_threads = get_nprocs(); 
    int tid = th_args->tid;
    
    arb_set(local_t, th_args->t);
    
    acb_onei(s);
    acb_mul_arb(s, s, local_t, MAX_PREC);
    arb_set_ui(onehalf, 1);
    arb_div_ui(onehalf, onehalf, 2, MAX_PREC);
    acb_add_arb(s, s, onehalf, MAX_PREC);  // s = 1/2 + it

    acb_onei(shifti);
    arb_set(shift, SHIFT);
    acb_mul_arb(shifti, shifti, shift, MAX_PREC);

    for(int i = 0; i < tid; i++)
	acb_add(s, s, shifti, MAX_PREC); 
   
    acb_mul_ui(shifti, shifti, num_threads, MAX_PREC); 
    
    int zeta_prec = 53; 

    for(int v = tid; v < th_args->k; v += num_threads){
	do{
	    acb_dirichlet_zeta(res, s, zeta_prec);
	    acb_get_real(realpart, res);
	    acb_get_imag(imagpart, res);
	}while(!correct_precision(realpart, &zeta_prec)
	       || !correct_precision(imagpart, &zeta_prec));

	zeta[v] = arb_get_d(realpart) + I*arb_get_d(imagpart);
	
	acb_add(s, s, shifti, MAX_PREC);
    }

    return NULL; 
    
}

double _Complex * stage0(arb_t t, int k){

    double _Complex * zeta = (double _Complex *) malloc(k*sizeof(double _Complex)); 
    
    int num_threads = get_nprocs();
    
    pthread_t threads[num_threads]; 
    struct thread_arguments th_args[num_threads]; 
    
    zeta = (double _Complex *) malloc(k*sizeof(double _Complex));
    for(int i = 0; i < k; i++) zeta[i] = 0; 
    
    for(int v = 0; v < num_threads; v++){
	th_args[v].tid = v;
	th_args[v].k = k;
	th_args[v].zeta = zeta;
	arb_init(th_args[v].t);
	arb_set(th_args[v].t, t);
	pthread_create(&threads[v], NULL, &simple_zeta_thread, &th_args[v]); 
    }
    
    for(int v = 0; v < num_threads; v++)
	pthread_join(threads[v], NULL); 

    return zeta; 
}
