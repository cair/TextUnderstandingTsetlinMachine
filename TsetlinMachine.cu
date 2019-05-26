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

This code implements the Tsetlin Machine from paper arXiv:1804.01508
https://arxiv.org/abs/1804.01508

*/

#include "TsetlinMachineKernels.cuh"
#include "TsetlinMachine.cuh"
#include "TsetlinMachineConfig.cuh"
#include "GPUConfig.cuh"

#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

/**********************************/
/***** Constructor/Destructor *****/
/**********************************/

TsetlinMachine::TsetlinMachine()
{
  cudaMallocManaged(&class_sum, sizeof(int));
  cudaMallocManaged(&ta_state, CLAUSES*LA_CHUNKS*STATE_BITS*sizeof(int));
  cudaMallocManaged(&clause_output, CLAUSES*sizeof(int));
  cudaMallocManaged(&clause_feedback, CLAUSES*sizeof(int));
  cudaMallocManaged(&all_exclude, CLAUSES*sizeof(int));
  //cudaMallocManaged(&la_feedback, LA_CHUNKS*sizeof(int));

  initialize();
}

TsetlinMachine::~TsetlinMachine()
{
  cudaFree(class_sum);
  cudaFree(ta_state);
  cudaFree(clause_output);
  cudaFree(clause_feedback); 
  cudaFree(all_exclude);
}

void TsetlinMachine::initialize()
{
  // Initialize Tsetlin Automata states
  for (int j = 0; j < CLAUSES; ++j) {
    for (int k = 0; k < LA_CHUNKS; ++k) {
      for (int b = 0; b < STATE_BITS-1; ++b) {
        int id = j*LA_CHUNKS*STATE_BITS + k*STATE_BITS + b;                   
        ta_state[id] = ~0;
      }

      int id = j*LA_CHUNKS*STATE_BITS + k*STATE_BITS + STATE_BITS-1;
      ta_state[id] = 0;
    }
  }
}

/****************************/
/***** Public Functions *****/
/****************************/

void TsetlinMachine::update(curandState *devStates, int *Xi, int target, float s)
{
  initialize_clause_output<<<GRID_SIZE,BLOCK_SIZE>>>(clause_output);
  cudaDeviceSynchronize();

  calculate_clause_output<<<GRID_SIZE,BLOCK_SIZE>>>(ta_state, clause_output, Xi);
  cudaDeviceSynchronize();

  int sum = 0;
  for (int clause = 0; clause < CLAUSES; ++clause) {
    int sign = 1 - 2 * (clause & 1);
    sum += sign * clause_output[clause];
  }

  if (sum > THRESHOLD) {
    sum = THRESHOLD;
  } else if (sum < -THRESHOLD) {
    sum = -THRESHOLD;
  }

  generate_clause_feedback<<<GRID_SIZE,BLOCK_SIZE>>>(devStates, clause_feedback, sum, target);
  cudaDeviceSynchronize();

  type_i_feedback<<<GRID_SIZE,BLOCK_SIZE>>>(devStates, ta_state, clause_feedback, clause_output,  Xi, s);
  cudaDeviceSynchronize();

  type_ii_feedback<<<GRID_SIZE,BLOCK_SIZE>>>(ta_state, clause_feedback, clause_output, Xi);
  cudaDeviceSynchronize();
}

int TsetlinMachine::get_state(int id)
{
  return ta_state[id];
}

int TsetlinMachine::score(int *Xi)
{
  initialize_clause_output_predict<<<GRID_SIZE,BLOCK_SIZE>>>(clause_output, all_exclude);
  cudaDeviceSynchronize();

  calculate_clause_output_predict<<<GRID_SIZE,BLOCK_SIZE>>>(ta_state, clause_output, all_exclude, Xi);
  cudaDeviceSynchronize();

  update_with_all_exclude<<<GRID_SIZE,BLOCK_SIZE>>>(clause_output, all_exclude);
  cudaDeviceSynchronize();

  int sum = 0;
  for (int clause = 0; clause < CLAUSES; ++clause) {
    int sign = 1 - 2 * (clause & 1);
    sum += sign * clause_output[clause];
  }

  if (sum > THRESHOLD) {
      sum = THRESHOLD;
  } else if (sum < -THRESHOLD) {
      sum = -THRESHOLD;
  }

  return sum;
}

