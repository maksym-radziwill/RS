#include <unistd.h>
#include "aux.h"
#include <curses.h>
#include <pthread.h>
#include "config.h"
#include <arf.h>
#include <sys/time.h>
#include "status_bar.h"

#ifdef HAS_MPI
#include <mpi.h>
#endif


void * status_bar_u(void * data){

    /* Allow the thread to be cancelled */ 
    
    int oldstate; // not used; 
    
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, &oldstate);
    pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, &oldstate);
    // Cancel only when reaching sleep
    
    int pthread_setcancelstate(int state, int *oldstate);
    
    /* Loading arguments */

    struct status_bar_args * args = (struct status_bar_args *) data; 
    
    fmpz_t global_n;
    fmpz_init(global_n);
    
    int world_size = args->world_size;
    int world_rank = args->world_rank;

    /* Getting the maximal n */ 
    
    fmpz_t maximal_n;
    fmpz_init(maximal_n);
    pthread_mutex_lock(args->n_mutex); 
    fmpz_set(maximal_n, *args->maximal_n);
    pthread_mutex_unlock(args->n_mutex); 

    /* Need an initial delay to gather data */ 

    struct timeval start, stop;
     
    /* Starting the status bar code */ 
    
    arf_t ratio; arf_init(ratio);
    double local_ratio[world_size];

    local_ratio[world_rank] = 0; 

    double global_ratio = 0; 

    int minutes = 0;
    int hours = 0; 
    
    clear();

    /* Compute the progress accomplished since the beginning of the countdown */ 

    gettimeofday(&start, NULL); 

    while(global_ratio < 0.999){

	/* Start the countdown and wait */ 

	sleep(1); 
	
	gettimeofday(&stop, NULL);
	
       	/* Compute the progress accomplished since the beginning of the countdown */ 
	
	pthread_mutex_lock(args->n_mutex);  
	fmpz_set(global_n, *args->global_n);
	pthread_mutex_unlock(args->n_mutex); 

	double micro_seconds = 1000*(stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec);
	
	arf_set_fmpz(ratio, global_n);
	arf_div_fmpz(ratio, ratio, maximal_n, 4*53, MPFR_RNDN);
	
	double local_value = arf_get_d(ratio, MPFR_RNDN); 
	
	/* Broadcast the progress and get the global average */

#ifdef HAS_MPI
        MPI_Allreduce(&local_value, &global_ratio, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      	MPI_Allgather(&local_value, 1, MPI_DOUBLE, local_ratio, 1, MPI_DOUBLE, MPI_COMM_WORLD);
#else
	global_ratio = local_value;
	local_ratio[world_rank] = local_value;
#endif

	global_ratio /= world_size;

	double multiplier = 1 / global_ratio;
	double seconds_remaining = (1 - global_ratio)*multiplier*((double) micro_seconds) / 1000; // time remaining 
	int seconds = (int) seconds_remaining % 60; 
	minutes = (int) seconds_remaining / 60;
	hours = minutes/60;
	minutes = minutes % 60;
	
	if((world_rank == 0)){
	    move(0,0);
	    refresh();
	    printw("%s\n", args->str);
	    
	    for(int i = 0; i < world_size; i++){
		move(i + 1, 0);
		refresh();
		printw("%s%d [%.1f%%]\n", "H", i, local_ratio[i]*100); 
	    }
	    
	    move(world_size + 1, 0);
	    refresh();

	    printw("%s %02dh %02dm %02ds\n", "ETA", hours, minutes, seconds);
	    
	    refresh();
	}

    }

    return NULL; 
    
}
