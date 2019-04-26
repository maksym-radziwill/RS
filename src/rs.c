
#include <getopt.h>
#include <arb.h>
#include <stdio.h>
#include <unistd.h>
#include <curses.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <acb.h>
#include <acb_dirichlet.h>
#include <string.h>
#include <sys/sysinfo.h>
#include <signal.h>
#include "aux.h"
#include "cuda.h"
#include "config.h"

#ifdef HAS_MPI
#include <mpi.h>
#endif


/* TODO: Split this post-processing into threads to speed it up for small values */ 

double _Complex * stage0(arb_t t, int k); 
double _Complex * stage1(arb_t t, int k, int world_rank, int world_size); 
double _Complex * stage2(arb_t t, int k, int world_rank, int world_size);

arb_t SHIFT; 

/* This is to handle window resize */ 

void do_resize( int sig ){
    if(sig == SIGWINCH)
	clearok(stdscr, TRUE); 
}

int silent_flag = 0;
int nocache_flag = 0; 
char * filename = NULL; 

int main(int argc, char ** argv){

    /* First handle the first most important and only non-optional argument */

    if(argc < 2){
	printf("Usage: %s t [--N k] [--shift delta] [--silent] [--nocache] [--filename output]\n", argv[0]);
	exit(0); 
    }
    
    arb_t t; arb_init(t);

    /* The precision here will limit the precision later, so it's important that
     * it is set to something unrealistically high */
    
    if(arb_set_str(t, argv[1], MAX_PREC)){
	fprintf(stderr, "Unable to parse the numerical value of t\n");
	exit(-1); 
    }

    arb_abs(t, t); 

    /* Collect the number of digits of t in base 10 */ 
    
    arb_init(SHIFT);
    arb_set_str(SHIFT, "0.01", MAX_PREC); 

    int k = 10 + 1; 
    
    /* Parse the optional arguments */ 

    int c; 
    
    while (1){
	static struct option long_options[] =
	    {
		{"silent",    no_argument,    &silent_flag, 1},
		{"nocache",   no_argument,    &nocache_flag, 1},
		{"shift",  required_argument,       0, 's'},
		{"filename",  required_argument, 0, 'f'},
		{"N",  required_argument, 0, 'N'},
		{0, 0, 0, 0}
	    };
	
	/* getopt_long stores the option index here. */
	int option_index = 0;
	
	c = getopt_long (argc, argv, "Sns:f:N:", long_options, &option_index);
	
	/* Detect the end of the options. */
	if (c == -1) break;
	
	switch (c){
	case 0: break;
	case 's':
	    arb_init(SHIFT);
	    arb_set_str(SHIFT, optarg, MAX_PREC); 
	    break;
	    
	case 'f':
	    filename = optarg; 
	    break;
	    
	case 'N':
	    k = atoi(optarg) + 1;
	    if(k <= 1) k = 2; 
	    break;
	    
	default:
	    break;
	}
    }

    arb_t digit; arb_init(digit);
    arb_log(digit, t, MAX_PREC);
    int digits = (int) arb_get_d(digit); 

    double _Complex * zeta;
    
    if(digits <= 16){ 
	
	/* If the t value is too small then use trivial algorithm */ 

	zeta = stage0(t, k); 

    }else{

#ifdef HAS_MPI
	MPI_Init(&argc, &argv);
#endif
	
	/* Curses initialization stuff */
	/* For bad reasons it's better to intialize after MPI */
	
	if(!silent_flag){
	    initscr(); noecho(); cbreak();
	    signal(SIGWINCH, do_resize);
	}
	
	int world_rank = 0; 
	int world_size = 1;

#ifdef HAS_MPI	
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); 
	MPI_Comm_size(MPI_COMM_WORLD, &world_size); 
#endif
	
	/* Stage 1 Computations */ 
	
	double _Complex * local_result_stage1 = (double _Complex *) malloc(k*sizeof(double _Complex)); 
	double _Complex * global_result_stage1 = (double _Complex *) malloc(k*sizeof(double _Complex));
	
	local_result_stage1 = stage1(t, k, world_rank, world_size); 

	/* Stage 2 computations */ 
	
	double _Complex * local_result_stage2 = (double _Complex *) malloc(k*sizeof(double _Complex));
	double _Complex * global_result_stage2 = (double _Complex *) malloc(k*sizeof(double _Complex)); 
       	
	local_result_stage2 = stage2(t, k, world_rank, world_size); 

#ifdef HAS_MPI	
	MPI_Reduce(local_result_stage1, global_result_stage1, k, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD); 
	MPI_Reduce(local_result_stage2, global_result_stage2, k, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD); 

	MPI_Finalize();
#else
	global_result_stage1 = local_result_stage1;
	global_result_stage2 = local_result_stage2; 
#endif
	
	if(world_rank != 0) exit(0); 
	
	/* Gather results */
	
	zeta = compute_zeta(global_result_stage1, global_result_stage2, t, k);
    }

    if(!silent_flag)
	endwin(); 
    
    double * HardyZ = compute_Z(zeta, t, k); 

    /* Output the data */

    FILE * fp;
    if(filename != NULL)
	fp = fopen (filename ,"a");
    else
	fp = stdout; 

    if(fp == NULL) fp = stdout; 

    for(int v = 0; v < k - 1; v++){
	fprintf(fp, "t              = ");
	arb_fprintn(fp, t, 35,  ARB_STR_NO_RADIUS); 
	fprintf(fp, "\nzeta(1/2 + it) = %.15f + %.15f * I\n"
		    "Z(t)           = %.15f\n"
		    "---------------------------------------------------\n", 
		creal(zeta[v]), cimag(zeta[v]), HardyZ[v]);
	
	arb_add(t, t, SHIFT, MAX_PREC);
    }
    
}
