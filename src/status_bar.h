struct status_bar_args {
    fmpz_t * global_n;
    fmpz_t * maximal_n; 
    int world_size;
    int world_rank;
    pthread_mutex_t * n_mutex; 
    char * str; 
};

void * status_bar_u(void * data);
