/*

Copyright (c) 2019 Ole-Christoffer Granmo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This code implements a multiclass version of the Tsetlin Machine from paper arXiv:1804.01508
https://arxiv.org/abs/1804.01508

*/


#include "TsetlinMachineConfig.cuh"
#include "TsetlinMachine.cuh"

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

// Increment the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
__device__ inline void inc(int *ta_state, int clause, int chunk, unsigned int active)
{
  unsigned int carry, carry_next;

  int id = clause*LA_CHUNKS*STATE_BITS + chunk*STATE_BITS;

  carry = active;
  for (int b = 0; b < STATE_BITS; ++b) {
    if (carry == 0)
      break;

    carry_next = ta_state[id + b] & carry; // Sets carry bits (overflow) passing on to next bit
    ta_state[id + b] = ta_state[id + b] ^ carry; // Performs increments with XOR
    carry = carry_next;
  }

  if (carry > 0) {
    for (int b = 0; b < STATE_BITS; ++b) {
      ta_state[id + b] |= carry;
    }
  }   
}

// Decrement the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
__device__ inline void dec(int *ta_state, int clause, int chunk, unsigned int active)
{
  unsigned int carry, carry_next;

  int id = clause*LA_CHUNKS*STATE_BITS + chunk*STATE_BITS;

  carry = active;
  for (int b = 0; b < STATE_BITS; ++b) {
    if (carry == 0)
      break;

    carry_next = (~ta_state[id + b]) & carry; // Sets carry bits (overflow) passing on to next bit
    ta_state[id + b] = ta_state[id + b] ^ carry; // Performs increments with XOR
    carry = carry_next;
  }

  if (carry > 0) {
    for (int b = 0; b < STATE_BITS; ++b) {
      ta_state[id + b] &= ~carry;
    }
  } 
}

__global__ void type_i_feedback(curandState *state, int *ta_state, int *clause_feedback, int *clause_output, int *Xi, float s)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  /* Copy state to local memory for efficiency */  
  curandState localState = state[index];

  for (int i = index; i < CLAUSES*LA_CHUNKS; i += stride) {
    int clause = i / LA_CHUNKS;

    if (clause_feedback[clause] != 1) {
      continue;
    }

    int la_chunk = i % LA_CHUNKS;

    // Generate random bit values

    int la_feedback;
    for (int b = 0; b < INT_SIZE; ++b) {
      if  (curand_uniform(&localState) <= 1.0/S) {
        la_feedback |= (1 << b);
      } else {
        la_feedback &= ~(1 << b);
      }
    }

    if (clause_output[clause]) {
      #ifdef BOOST_TRUE_POSITIVE_FEEDBACK
        inc(ta_state, clause, la_chunk, Xi[la_chunk]);
      #else
        inc(ta_state, clause, la_chunk, Xi[la_chunk] & (~la_feedback));
      #endif
      
      dec(ta_state, clause, la_chunk, (~Xi[la_chunk]) & la_feedback);
    } else {
      dec(ta_state, clause, la_chunk, la_feedback);
    }
  }

  state[index] = localState;
}

__global__ void type_ii_feedback(int *ta_state, int *clause_feedback, int *clause_output, int *Xi)
{

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < CLAUSES*LA_CHUNKS; i += stride) {
    int clause = i / LA_CHUNKS;

    if (clause_feedback[clause] != -1 || clause_output[clause] == 0) {
      continue;
    }

    int la_chunk = i % LA_CHUNKS;

    int id = clause*LA_CHUNKS*STATE_BITS + la_chunk*STATE_BITS + STATE_BITS - 1;

    inc(ta_state, clause, la_chunk, (~Xi[la_chunk]) & (~ta_state[id]));
  }
}

__global__ void generate_clause_feedback(curandState *state, int *clause_feedback, int class_sum, int target)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  /* Copy state to local memory for efficiency */  
  curandState localState = state[index];

  for (int j = index; j < CLAUSES; j += stride) {
    int sign =  1 - 2 * (j & 1);

    if (target) {
      if (curand_uniform(&localState) > (1.0/(THRESHOLD*2))*(THRESHOLD - class_sum)) {
        clause_feedback[j] = 0;
      } else {
        clause_feedback[j] = sign;
      }
    } else {
      if (curand_uniform(&localState) > (1.0/(THRESHOLD*2))*(THRESHOLD + class_sum)) {
        clause_feedback[j] = 0;
      } else {
        clause_feedback[j] = -1*sign;
      }
    }
  }

  state[index] = localState;
}

__global__ void initialize_clause_output(int *clause_output)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // Initialize clause output
  for (int j = index; j < CLAUSES; j += stride) {
    clause_output[j] = 1;
  }
}

__global__ void calculate_clause_output(int *ta_state, int *clause_output, int *Xi)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < CLAUSES*LA_CHUNKS; i += stride) {
    int clause = i / LA_CHUNKS;
    int la_chunk = i % LA_CHUNKS;

    int id = clause*LA_CHUNKS*STATE_BITS + la_chunk*STATE_BITS + STATE_BITS - 1;
    if (la_chunk < LA_CHUNKS-1 && ((ta_state[id] & Xi[la_chunk]) != ta_state[id])) {
      clause_output[clause] = 0;
    } else if (la_chunk == LA_CHUNKS-1 && ((ta_state[id] & Xi[LA_CHUNKS-1] & FILTER) != (ta_state[id] & FILTER))) {
      clause_output[clause] = 0;
    }
  }
}

__global__ void initialize_clause_output_predict(int *clause_output, int *all_exclude)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // Initialize clause output
  for (int j = index; j < CLAUSES; j += stride) {
    clause_output[j] = 1;
    all_exclude[j] = 1;
  }
}

__global__ void calculate_clause_output_predict(int *ta_state, int *clause_output, int *all_exclude, int *Xi)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < CLAUSES*LA_CHUNKS; i += stride) {
    int clause = i / LA_CHUNKS;
    int la_chunk = i % LA_CHUNKS;

    int id = clause*LA_CHUNKS*STATE_BITS + la_chunk*STATE_BITS + STATE_BITS - 1;

    if ((la_chunk < LA_CHUNKS - 1) && ((ta_state[id] & Xi[la_chunk]) != ta_state[id])) {
      clause_output[clause] = 0;
    } else if ((la_chunk == LA_CHUNKS - 1) && ((ta_state[id] & Xi[LA_CHUNKS-1] & FILTER) != (ta_state[id] & FILTER))) {
      clause_output[clause] = 0;
    }

    if ((la_chunk < LA_CHUNKS - 1) && ((ta_state[id] & Xi[la_chunk]) > 0)) {
      all_exclude[clause] = 0;
    } else if ((la_chunk == LA_CHUNKS - 1) && ((ta_state[id] & Xi[LA_CHUNKS-1] & FILTER) > 0)) {
      all_exclude[clause] = 0;
    }
  }
}

__global__ void update_with_all_exclude(int *clause_output, int *all_exclude)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // Initialize clause output
  for (int j = index; j < CLAUSES; j += stride) {
    if (all_exclude[j] == 1) {
      clause_output[j] = 0;
    }
  }
}
