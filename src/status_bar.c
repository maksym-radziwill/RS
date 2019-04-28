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

static double local_value; 

void * status_bar_u(void * data){

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

    /* Starting the status bar code */ 
    
    arf_t ratio; arf_init(ratio);
    double local_ratio[world_size];

    local_ratio[world_rank] = 0; 

    double global_ratio = 0; 
    double old_global_ratio = 0; 
    
    int minutes = 0;
    int hours = 0; 
    
    clear();

    while(1){

      /* Check if all hosts have returned */ 
      
#ifdef HAS_MPI
      MPI_Barrier(MPI_COMM_WORLD); 
#endif
      
      double counter = 0;
      for(int i = 0; i < world_size; i++)
	counter += local_ratio[i];       
      
      if((int) counter == world_size) break; 
      
      /* Compute the progress accomplished since the beginning of the countdown */ 
      
      pthread_mutex_lock(args->n_mutex);  
      fmpz_set(global_n, *args->global_n);
      pthread_mutex_unlock(args->n_mutex); 
      
      arf_set_fmpz(ratio, global_n);
      arf_div_fmpz(ratio, ratio, maximal_n, 4*53, MPFR_RNDN);
      
      local_value = arf_get_d(ratio, MPFR_RNDN); 
      
      /* Share individual progress and get the global average */

#ifdef HAS_MPI
        MPI_Reduce(&local_value, &global_ratio, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Allgather(&local_value, 1, MPI_DOUBLE, local_ratio, 1, MPI_DOUBLE, MPI_COMM_WORLD);
#else
	global_ratio = local_value;
	local_ratio[world_rank] = local_value;
#endif
	
	global_ratio /= world_size;
		
	if((world_rank == 0) && old_global_ratio < global_ratio - 0.01){
	    move(0,0);
	    refresh();
	    printw("%s ", args->str);
	    printw("[%.1f%%]\n", global_ratio*100); 
	    for(int i = 0; i < world_size; i++){
		move(i + 2, 0);
		refresh();
		printw("%s%d [%.1f%%]\n", "H", i, local_ratio[i]*100); 
	    }
	    refresh();

	    old_global_ratio = global_ratio;
	    usleep(100000); 
	}
	
    }

    return NULL;     
}
