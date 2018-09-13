#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

__global__ void setup_kernel(curandState *state);


__global__ void type_i_feedback(curandState *state, int *ta_state, int *clause_feedback, int *clause_output, int *Xi, float s);


__global__ void type_ii_feedback(int *ta_state, int *clause_feedback, int *clause_output, int *Xi);


/* Sum up the votes for each class (this is the multiclass version of the Tsetlin Machine) */
__global__ void sum_up_class_votes(int *clause_output, int *sum);


/* Sum up the votes for each class (this is the multiclass version of the Tsetlin Machine) */
__global__ void generate_clause_feedback(curandState *state, int *clause_feedback, int *class_sum, int target);

__global__ void initialize_clause_output(int *clause_output);

__global__ void calculate_clause_output(int *ta_state, int *clause_output, int *Xi);


__global__ void initialize_clause_output_predict(int *clause_output, int *all_exclude);


__global__ void calculate_clause_output_predict(int *ta_state, int *clause_output, int *all_exclude, int *Xi);


__global__ void update_with_all_exclude(int *clause_output, int *all_exclude);