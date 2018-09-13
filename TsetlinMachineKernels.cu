#include "TsetlinMachineConfig.cuh"

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

__device__ int inline action(int state)
{
    if (state <= NUMBER_OF_STATES)
      return 0;
    else
      return 1;
}

__global__ void type_i_feedback(curandState *state, int *ta_state, int *clause_feedback, int *clause_output, int *Xi, float s)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  /* Copy state to local memory for efficiency */  
  curandState localState = state[index];

  for (int i = index; i < CLAUSES*FEATURES; i += stride) {
    int clause = i / FEATURES;
    int feature = i % FEATURES;

    int output = clause_output[clause];

    if (clause_feedback[clause] != 1) {
      continue;
    }

    if (output == 0) {                      
      if (curand_uniform(&localState) <= 1.0/s) {
        if (ta_state[i*2] > 1) {
          ta_state[i*2] -= 1;
        }
      }

      if (curand_uniform(&localState) <= 1.0/s) {
        if (ta_state[i*2+1] > 1) {
          ta_state[i*2+1] -= 1;
        }
      }
    } else if (output == 1) {          
      if (Xi[feature] == 1) {
        if (BOOST_TRUE_POSITIVE_FEEDBACK == 1 || curand_uniform(&localState) <= (s-1)/s) {
          if (ta_state[i*2] < NUMBER_OF_STATES*2) {
            ta_state[i*2] += 1;
          }
        }

        if (curand_uniform(&localState) <= 1.0/s) {
          if (ta_state[i*2+1] > 1) {
            ta_state[i*2+1] -= 1;
          }
        }
      } else if (Xi[feature] == 0) {
        if (BOOST_TRUE_POSITIVE_FEEDBACK == 1 || curand_uniform(&localState) <= (s-1)/s){
          if (ta_state[i*2+1] < NUMBER_OF_STATES*2) {
            ta_state[i*2+1] += 1;
          }
        }

        if (curand_uniform(&localState) <= 1.0/s) {
          if (ta_state[i*2] > 1) {
            ta_state[i*2] -= 1;
          }
        }
      }
    }
  }

  state[index] = localState;
}

__global__ void type_ii_feedback(int *ta_state, int *clause_feedback, int *clause_output, int *Xi)
{

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int action_include;
  int action_include_negated;

  for (int i = index; i < CLAUSES*FEATURES; i += stride) {
    int clause = i / FEATURES;
    int feature = i % FEATURES;

    if (clause_feedback[clause] != -1 || clause_output[clause] == 0) {
      continue;
    }

    action_include = action(ta_state[i*2]);
    action_include_negated = action(ta_state[i*2+1]);

    if (Xi[feature] == 0) {
      if (action_include == 0 && ta_state[i*2] < NUMBER_OF_STATES*2) {
        ta_state[i*2] += 1;
      }
    } else if (Xi[feature] == 1) {
      if (action_include_negated == 0 && ta_state[i*2+1] < NUMBER_OF_STATES*2) {
        ta_state[i*2+1] += 1;
      }
    }
  }
}

/* Sum up the votes for each class (this is the multiclass version of the Tsetlin Machine) */
__global__ void sum_up_class_votes(int *clause_output, int *sum)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int local_sum = 0;
  for (int j = index; j < CLAUSES; j += stride) {
    int sign = 1 - 2 * (j & 1);

    local_sum += sign * clause_output[j];
  }

  atomicAdd(sum, local_sum);
}

/* Sum up the votes for each class (this is the multiclass version of the Tsetlin Machine) */
__global__ void generate_clause_feedback(curandState *state, int *clause_feedback, int *class_sum, int target)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  /* Copy state to local memory for efficiency */  
  curandState localState = state[index];

  for (int j = index; j < CLAUSES; j += stride) {
    int sign =  1 - 2 * (j & 1);

    if (target) {
      if (curand_uniform(&localState) > (1.0/(THRESHOLD*2))*(THRESHOLD - *class_sum)) {
        clause_feedback[j] = 0;
      } else {
        clause_feedback[j] = sign;
      }
    } else {
      if (curand_uniform(&localState) > (1.0/(THRESHOLD*2))*(THRESHOLD + *class_sum)) {
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

  int action_include, action_include_negated;

  for (int i = index; i < CLAUSES*FEATURES; i += stride) {
    int clause = i / FEATURES;
    int feature = i % FEATURES;

    action_include = action(ta_state[i*2]);
    action_include_negated = action(ta_state[i*2+1]);

    if ((action_include == 1 && Xi[feature] == 0) || (action_include_negated == 1 && Xi[feature] == 1)) {
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

  int action_include, action_include_negated;

  for (int i = index; i < CLAUSES*FEATURES; i += stride) {
    int clause = i / FEATURES;
    int feature = i % FEATURES;

    action_include = action(ta_state[i*2]);
    action_include_negated = action(ta_state[i*2+1]);

    if (action_include == 1 || action_include_negated == 1) {
      all_exclude[clause] = 0;
    }

    if ((action_include == 1 && Xi[feature] == 0) || (action_include_negated == 1 && Xi[feature] == 1)) {
      clause_output[clause] = 0;
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