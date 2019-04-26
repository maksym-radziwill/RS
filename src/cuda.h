#include <cuda_runtime.h>
#include "config.h"

// Might want to keep under 128 bits

#define LOG_MAX 8
#define SQRT_MAX 4
#define ALIGN_BLOCK 128 * ( 1 + (8*LOG_MAX + 8*SQRT_MAX + 4 + 28) / 128)

#define BLOCK_SIZE 8*(ALIGN_BLOCK / 32)
#define MEM_BLOCK 2048

struct computation_data {
    double log_constants[LOG_MAX]; // + 64
    double sqrt_constants[SQRT_MAX];  // + 32
    unsigned long len; // + 4
} __align__ (ALIGN_BLOCK); // = 100

struct computation_results {
    double cos_results;
    double sin_results; 
} __align__ (16);

