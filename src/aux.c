#include <arb.h>
#include <acb.h>
#include <acb_dirichlet.h>
#include <flint/fmpz.h>
#include <pthread.h>
#include "md5.h"
#include "aux.h"
#include "cuda.h"



int gcd(int m, int n)
{
    int tmp;
    while(m) { tmp = m; m = n % m; n = tmp; }
    return n;
}

int lcm(int m, int n)
{
    return m / gcd(m, n) * n;
}

char *str2md5(const char *str, int length) {
    int n;
    MD5_CTX c;
    unsigned char digest[16];
    char *out = (char*)malloc(33);

    MD5_Init(&c);

    while (length > 0) {
	if (length > 512) {
	    MD5_Update(&c, str, 512);
	} else {
	    MD5_Update(&c, str, length);
	}
	length -= 512;
	str += 512;
    }

    MD5_Final(digest, &c);

    for (n = 0; n < 16; ++n) {
	snprintf(&(out[n*2]), 16*2, "%02x", (unsigned int)digest[n]);
    }

    return out;
}

char * hash_name (arb_t local_t, int world_rank, int world_size, int k, int stage){
    char * str = arb_get_str(local_t, MAX_PREC, ARB_STR_NO_RADIUS);
    char * new_str = (char *) malloc(strlen(str) + 5*16 + 2*MAX_PREC + 1024);
    
    sprintf(new_str, "%s,%d,%d,%d,%d,%s",
	    arb_get_str(local_t, MAX_PREC, ARB_STR_NO_RADIUS),
	    world_rank, world_size, k, stage,
	    arb_get_str(SHIFT, MAX_PREC, ARB_STR_NO_RADIUS));

    char * hashed_str = str2md5(new_str, strlen(new_str));
    char * final_str = (char *) malloc(strlen(hashed_str) + strlen(".data"));

    sprintf(final_str, ".data%s", hashed_str); 

    free(hashed_str);
    free(new_str);
    free(str);

    return final_str; 
}

int load_data(arb_t local_t, int k, double _Complex * res, int world_rnk, int world_sz, int stage, fmpz_t global_n){

    char * name = hash_name(local_t, world_rnk, world_sz, k, stage);

    FILE * fp = fopen(name, "r");
    if(!fp) return 0; // File does not exist
    
    int length;
    if(fread(&length, sizeof(int), 1, fp) != 1){
	fprintf(stderr, "Unable to read the cache file\n");
	return 0; 
    }
    
    char * global_n_str = (char *) malloc((length + 10)*sizeof(char));
    if(fread(global_n_str, sizeof(char), length, fp) != length){
	fprintf(stderr, "Unable to read the cache file\n");
	return 0; 
    }
    
    global_n_str[length] = 0;
    fmpz_set_str(global_n, global_n_str, 10);

    if(fread(res, sizeof(double _Complex), k, fp) != k){
	fprintf(stderr, "Unable to read the cache file\n");
	for(int i = 0; i < k; i++) res[i] = 0; // A precaution
	return 0; 
    }

    free(name); 
    free(global_n_str); 

    return 1;
}

void save_data(arb_t local_t, int k, double _Complex * res, int world_rnk, int world_sz, int stage, fmpz_t global_n){

    char * name = hash_name(local_t, world_rnk, world_sz, k, stage); // Hash the name given the information we have

    FILE * fp = fopen(name, "wb"); // Create file or discard existing one

    if(fp == NULL){
	fprintf(stderr, "Unable to open file %s\n", name); 
	return;
    }
    
    /* Save the data associated to the unique hash */ 
    
    char * global_n_str = fmpz_get_str(NULL, 10, global_n);
    int length = strlen(global_n_str);

    fwrite(&length, sizeof(int), 1, fp);
    fwrite(global_n_str, sizeof(char), strlen(global_n_str), fp);
    fwrite(res, sizeof(double _Complex), k, fp); 

    fclose(fp); 

    free(global_n_str);
    free(name);     
}

void adjust_prec(int * prec){
    *prec += 1;
}

int correct_precision(arb_t temp, int * prec){
    if(arb_can_round_mpfr(temp, 80, MPFR_RNDN))
	return 1;
    else{
	adjust_prec(prec);
	return 0;
    }
}

int half_precision(arb_t temp, int * prec){
    if(arb_can_round_mpfr(temp, 40, MPFR_RNDN))
	return 1;
    else{
	adjust_prec(prec);
	return 0;
    }
}


int factorial(int k){
    static int table[12] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800};

    if((k >= 0) && (k <= 12)){
        return table[k];
    }else{
        fprintf(stderr, "Requested a non-implemented factorial\n");
        exit(-1);
    }
}

int double_factorial(int k){
    static int table[11] = {1, 1, 3, 15, 105, 945, 10395, 135135, 2027025, 34459425, 654729075};
    if((k >= 0) && (k <= 11)){
        return table[k];
    }else{
        fprintf(stderr, "Requested a non-implemented double-factorial\n");
        exit(-1);
    }
}


/* Sets r = \floor t^{1/pow} */

void fmpz_floor_pow(fmpz_t r, arb_t t, int pow){

    arb_t temp;
    arb_init(temp);

    int prec = 4*53;

    arb_root_ui(temp, t, pow, prec);
    arb_floor(temp, temp, prec);

    if(!arb_get_unique_fmpz(r, temp)){ // r1 = \floor t^{1/4}
        fprintf(stderr, "Failed converting to fmpz\n");
        exit(-1);
    }

    arb_clear(temp);

    return;
}


double _Complex * compute_zeta(double _Complex * result_stage1, double _Complex * result_stage2, arb_t t, int k){
    double _Complex * zeta = (double _Complex *) malloc(k*sizeof(double _Complex));

    acb_t correction; acb_init(correction); 
    acb_t s; acb_init(s);
    arb_t onehalf; arb_init(onehalf);
    arb_t realpart; arb_init(realpart);
    arb_t imagpart; arb_init(imagpart); 
    arb_t local_t; arb_init(local_t); 
    arb_t shift; arb_init(shift); 
    acb_t shifti; acb_init(shifti); 
    
    arb_set(local_t, t);
    arb_set(shift, SHIFT);

    acb_onei(shifti);
    acb_mul_arb(shifti, shifti, shift, MAX_PREC);
    
    acb_onei(s);
    acb_mul_arb(s, s, local_t, MAX_PREC);
    arb_set_ui(onehalf, 1);
    arb_div_ui(onehalf, onehalf, 2, MAX_PREC);
    acb_add_arb(s, s, onehalf, MAX_PREC);  // s = 1/2 + it

    int theta_prec = 53; 
    int correction_prec = 53; 
    
    for(int v = 0; v < k; v++){
	
	do{
	    acb_dirichlet_zeta_rs_correction(correction, s, -1, correction_prec);
	    acb_get_real(realpart, correction);
	    acb_get_imag(imagpart, correction);
	}while(!correct_precision(realpart, &correction_prec) ||
	       !correct_precision(imagpart, &correction_prec)); 

	double _Complex correction_term = arb_get_d(realpart) + I*arb_get_d(imagpart);

	zeta[v] = result_stage1[v] + result_stage2[v] + correction_term;

	double _Complex theta2 = theta_phase(local_t, &theta_prec, 2); 
	
	zeta[v] += conj(theta2)*conj(result_stage1[v] + result_stage2[v] + correction_term); 

	acb_add(s, s, shifti, MAX_PREC);
	arb_add(local_t, local_t, shift, MAX_PREC); 
    }

    return zeta; 

}

double * compute_Z(double _Complex * zeta, arb_t t, int k){

    arb_t local_t; arb_init(local_t);
    arb_set(local_t, t); 

    arb_t shift; arb_init(shift);
    arb_set(shift, SHIFT);
    
    double * HardyZ = (double *) malloc(k*sizeof(double _Complex)); 

    int theta_prec = 53; 
    
    for(int v = 0; v < k; v++){
	double _Complex theta = theta_phase(local_t, &theta_prec, 1);
	HardyZ[v] = creal(theta*(zeta[v])); 
	arb_add(local_t, local_t, shift, MAX_PREC); 
    }

    return HardyZ;

}


/* Computes \theta(t) up to correct double precision */
/* This function might be no longer necessary */ 

void theta(arb_t output, arb_t t, int prec){
    acb_t theta, complex_t;

    acb_init(theta); acb_init(complex_t);
    acb_set_arb(complex_t, t);
    
    dirichlet_group_t G;
    dirichlet_char_t chi;
    dirichlet_group_init(G, 1);
    dirichlet_char_init(chi, G);
    dirichlet_char_log(chi, G, 1);
    
    // Not sure what the first len terms in the Taylor expansion refer to
    // in the documentation -- but this seems to work with the parameter 1
    // and nothing else.
    acb_dirichlet_hardy_theta(theta, complex_t, G, chi, 1, prec);
    acb_get_real(output, theta);
	
    acb_clear(theta);
    acb_clear(complex_t);
    dirichlet_group_clear(G);
    dirichlet_char_clear(chi);
}

/* Get theta phase with correct double precision */
/* Use prec as a dummy variable that we keep in memory
   to avoid recomputation of the precision */ 


double _Complex theta_phase(arb_t t, int * prec, int k){
    return multiply_by_theta_phase(1,0,t, prec, k); 
}



/* Multiplies (result_cos + i result_sin) by a theta phase and returns the resulting complex number */
/* We keep the imaginary part for error checking -- if the computation was correct this has to be 
   close to zero */ 

double _Complex multiply_by_theta_phase (double result_cos, double result_sin, arb_t t, int * prec, int k){

    arb_t output; arb_init(output);

    /* Compute cos of the theta phase */ 

    acb_t theta, complex_t;
    
    acb_init(theta); acb_init(complex_t);
    acb_set_arb(complex_t, t);
    
    dirichlet_group_t G;
    dirichlet_char_t chi;
    dirichlet_group_init(G, 1);
    dirichlet_char_init(chi, G);
    dirichlet_char_log(chi, G, 1);    

    do{
	// Not sure what the first len terms in the Taylor expansion refer to
	// in the documentation -- but this seems to work with the parameter 1
	// and nothing else.
	acb_dirichlet_hardy_theta(theta, complex_t, G, chi, 1, *prec);
	acb_mul_ui(theta, theta, k, *prec); 
	acb_get_real(output, theta);
	arb_cos(output, output, *prec);
    }while(!correct_precision(output, prec));

    double theta_phase_cos = arb_get_d(output);

    /* Compute sin of the theta phase */ 
    
    do{
	// Not sure what the first len terms in the Taylor expansion refer to
	// in the documentation -- but this seems to work with the parameter 1
	// and nothing else.
	acb_dirichlet_hardy_theta(theta, complex_t, G, chi, 1, *prec);
	acb_mul_ui(theta, theta, k, *prec); 
	acb_get_real(output, theta);
	arb_sin(output, output, *prec);
    }while(!correct_precision(output, prec));

    double theta_phase_sin = arb_get_d(output);

    arb_clear(output);

    /* Return final result multiplied by the theta phase */ 

    acb_clear(theta);
    acb_clear(complex_t);
    dirichlet_group_clear(G);
    dirichlet_char_clear(chi);
    
    return (theta_phase_cos * result_cos - theta_phase_sin * result_sin) +
	I * (theta_phase_cos * result_sin + theta_phase_sin * result_cos); 

}


void clear_temp(arb_t * temp, int temp_size){

    for(int i = 0; i < temp_size; i++)
        arb_clear(temp[i]);

    free(temp);
}

arb_t * init_temp(int num){
    arb_t * temp = (arb_t *) malloc(num*sizeof(arb_t));

    if(temp == NULL){
        fprintf(stderr, "Unable to allocate %lu bytes\n", num*sizeof(arb_t));
        exit(-1);
    }

    for(int i = 0; i < num; i++){
      arb_init(temp[i]);
    }
    
    return temp;
}

void riemann_siegel_length(fmpz_t output, arb_t t){

    arb_t * temp = init_temp(1);

    int prec = 4*53;

    arb_const_pi(temp[0], prec);
    arb_mul_ui(temp[0], temp[0], 2, prec);
    arb_inv(temp[0], temp[0], prec);
    arb_mul(temp[0], temp[0], t, prec);
    arb_sqrt(temp[0], temp[0], prec);

    arb_floor(temp[0], temp[0], prec);

    if(!arb_get_unique_fmpz(output, temp[0])){
        fprintf(stderr,"Failed to convert length to an integer. Aborting\n");
        exit(-1);
    }

    clear_temp(temp, 1);

}

void stage2_range(fmpz_t output, arb_t t){

    fmpz_t rs; fmpz_init(rs); 

    riemann_siegel_length(rs, t); 
    
    fmpz_floor_pow(output, t, 4);
    fmpz_mul_ui(output, output, 16); // 16 worked well here
    
    if(fmpz_cmp(output, rs) > 0){
	fmpz_set(output, rs);
	printf("Stage 1 bigger than stage2 \n"); 
    }
   
    //   fmpz_clear(rs);	

    return;
}

double arb_get_d(arb_t x){
    return arf_get_d(arb_midref(x),ARF_RND_DOWN);
}




int determine_sqrt_constants(fmpz_t n, arb_t t){
  int num = 4;

  if(num >= SQRT_MAX - 1){
    fprintf(stderr, "SQRT_MAX needs to be increased\n");
    exit(-1);
  }

  return num; // return k gives error of t^{-k/4}
}

int determine_log_constants_slow(fmpz_t n, arb_t t){
    arb_t * temp = init_temp(3);

    int prec = 53;

    arb_log_fmpz(temp[0], n, prec); // temp[0] = \log n
    arb_log(temp[1], t, prec); // temp[1] = \log t
    arb_div(temp[2], temp[1], temp[0], prec); // temp[2] = \log t / \log n
    arb_ceil(temp[2], temp[2], prec);

    int num = (int) arb_get_d(temp[2]) + 4; // + k instead of + 2 gives error of t^{-k/4}

    if(num >= LOG_MAX - 1){
      fprintf(stderr, "LOG_MAX needs to be increased\n");
      exit(-1);
    }

    clear_temp(temp, 3);

    return num;
}

void find_log_derivatives(arb_t output, arb_t x, int k, int prec){

    if(k == 0){
        arb_set(output, x);
        arb_add_ui(output, output, 1, prec);
        arb_log(output, output, prec); // output[0] = \log(1 + x)
        return;
    }

    if(k >= 1){
        arb_set(output, x);
        arb_add_ui(output, output, 1, prec);
        arb_inv(output, output, prec);
        arb_pow_ui(output, output, k, prec);
        arb_div_ui(output, output, k, prec);
        if(k % 2 == 0)
            arb_mul_si(output, output, -1, prec);
        return;
    }
}

void find_sqrt_derivatives(arb_t output, arb_t x, int k, int prec){

    arb_set(output, x);
    arb_add_ui(output, output, 1, prec);
    arb_sqrt(output, output, prec);
    arb_inv(output, output, prec); // output[0] = 1/\sqrt{1 + x}
    arb_pow_ui(output, output, 2*k + 1, prec);
    arb_div_ui(output, output, factorial(k), prec);
    arb_mul_ui(output, output, double_factorial(k), prec);
    arb_div_ui(output, output, (int) pow(2, k), prec);

    if(k % 2 == 1)
        arb_mul_si(output, output, -1, prec);

    return;

}

void arb_fmod(arb_t output, arb_t x, arb_t y, int prec){

    arb_abs(output, x);

    if(arb_le(output, y)){
        arb_set(output, x);
        return;
    }

    arb_div(output, x, y, prec); // output = x / y
    arb_floor(output, output, prec); // output = \floor {x / y}
    arb_mul(output, output, y, prec); // output = \floor {x / y} * y
    arb_sub(output, x, output, prec); // output = x - \floor {x / y} * y

}

pthread_attr_t thread_priority(int newprio){
    pthread_attr_t tattr; 
    struct sched_param param;

    /* initialized with default attributes */
    pthread_attr_init (&tattr);

    /* safe to get existing scheduling param */
    pthread_attr_getschedparam (&tattr, &param);

    /* set the priority; others are unchanged */
    param.sched_priority = newprio;
    
    /* setting the new scheduling param */
    pthread_attr_setschedparam (&tattr, &param);

    return tattr;
}
