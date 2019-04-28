/* We are allowed to pass 256 bytes to a CUDA kernel
   Of these 8 bytes are taken by two ints. 
   That leaves us with 31 double's. The define below
   delimite the maximal size of the array's 
   that will be passed to the CUDA kernel */ 

#include <cuda_runtime.h>
#include <complex.h>
#include <acb.h>
#include <string.h>
#include <pthread.h>

#define MAX_PREC 10*53 // Maximum precision of arb

extern arb_t SHIFT; 

extern int silent_flag;
extern int nocache_flag;

void clear_temp(arb_t * temp, int temp_size);
arb_t * init_temp(int num);
void riemann_siegel_length(fmpz_t output, arb_t t);
void stage2_range(fmpz_t output, arb_t t);
void theta(arb_t output, arb_t t, int prec);
double arb_get_d(arb_t x);
int determine_sqrt_constants(fmpz_t n, arb_t t);
int determine_log_constants_slow(fmpz_t n, arb_t t);
double _Complex multiply_by_theta_phase (double result_cos, double result_sin, arb_t t , int *, int);
double _Complex theta_phase(arb_t t , int *, int ); 
void find_log_derivatives(arb_t output, arb_t x, int k, int prec);
void find_sqrt_derivatives(arb_t output, arb_t x, int k, int prec);
void arb_fmod(arb_t output, arb_t x, arb_t y, int prec);
int correct_precision(arb_t temp, int * prec);
int half_precision(arb_t temp, int * prec);
double * compute_Nt(double _Complex * zeta, arb_t t, int k );
void acb_dirichlet_zeta_rs_correction(acb_t res, const acb_t s, slong K, slong prec);
int load_data(arb_t local_t, int k, double _Complex * res, int world_rnk, int world_sz, int stage, fmpz_t global_n);
void save_data(arb_t local_t, int k, double _Complex * res, int world_rnk, int world_sz, int stage, fmpz_t global_n);
double * compute_Z(double _Complex * zeta, arb_t t, int k);
double _Complex * compute_zeta(double _Complex * result_stage1, double _Complex * result_stage2, arb_t t, int k);
int gcd(int m, int n);
int lcm(int m, int n);
pthread_attr_t thread_priority(int); 
