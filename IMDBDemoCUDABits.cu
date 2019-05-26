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

#include <iostream>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "TsetlinMachineConfig.cuh"
#include "MultiClassTsetlinMachine.cuh"
#include "GPUConfig.cuh"

#define NUMBER_OF_TRAINING_EXAMPLES 25000
#define NUMBER_OF_TEST_EXAMPLES 25000

#define EXPERIMENTS 100
#define EPOCHS 200

#define DEVICE 0

int y_train[NUMBER_OF_TRAINING_EXAMPLES], y_test[NUMBER_OF_TEST_EXAMPLES];
int *X_train;
int *X_test;

void read_file(void)
{
  FILE * fp;
  char * line = NULL;
  size_t len = 0;

  const char *s = " ";
  char *token = NULL;

  // Training Dataset

  for (int i = 0; i < NUMBER_OF_TRAINING_EXAMPLES; i++) {
    for (int j = 0; j < LA_CHUNKS; j++) {
      X_train[i*LA_CHUNKS + j] = 0;
    }
  }

  fp = fopen("IMDBTrainingData.txt", "r");
  if (fp == NULL) {
    printf("Error opening\n");
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < NUMBER_OF_TRAINING_EXAMPLES; i++) {
    getline(&line, &len, fp);

    token = strtok(line, s);
    for (int j = 0; j < FEATURES; j++) {
      if (atoi(token) == 1) {
        int chunk_nr = j / INT_SIZE;
        int chunk_pos = j % INT_SIZE;
        X_train[i*LA_CHUNKS + chunk_nr] |= (1 << chunk_pos);
      } else {
        int chunk_nr = (j + FEATURES) / INT_SIZE;
        int chunk_pos = (j + FEATURES) % INT_SIZE;
        X_train[i*LA_CHUNKS + chunk_nr] |= (1 << chunk_pos);
      }
      token=strtok(NULL,s);
    }
    y_train[i] = atoi(token);
  }
  fclose(fp);

  // Test Dataset

  for (int i = 0; i < NUMBER_OF_TEST_EXAMPLES; i++) {
    for (int j = 0; j < LA_CHUNKS; j++) {
      X_test[i*LA_CHUNKS + j] = 0;
    }
  }

  fp = fopen("IMDBTestData.txt", "r");
  if (fp == NULL) {
    printf("Error opening\n");
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < NUMBER_OF_TEST_EXAMPLES; i++) {
    getline(&line, &len, fp);

    token = strtok(line, s);
    for (int j = 0; j < FEATURES; j++) {
      if (atoi(token) == 1) {
        int chunk_nr = j / INT_SIZE;
        int chunk_pos = j % INT_SIZE;
        X_test[i*LA_CHUNKS + chunk_nr] |= (1 << chunk_pos);
      } else {
        int chunk_nr = (j + FEATURES) / INT_SIZE;
        int chunk_pos = (j + FEATURES) % INT_SIZE;
        X_test[i*LA_CHUNKS + chunk_nr] |= (1 << chunk_pos);
      }
      token=strtok(NULL,s);
    }
    y_test[i] = atoi(token);
  }
  fclose(fp);
}

__global__ void setup_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    /* Each thread gets same seed, a different sequence 
       number, no offset */
    curand_init(1234, id, 0, &state[id]);
}

int main(void)
{
  FILE *fp;

  curandState *devStates;

  cudaSetDevice(DEVICE);

  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, DEVICE);

  printf("Num SMS: %d\n", numSMs);

  // Allocate Unified Memory – accessible from CPU or GPU

  cudaMallocManaged(&X_train, NUMBER_OF_TRAINING_EXAMPLES * LA_CHUNKS * sizeof(int));
  cudaMallocManaged(&X_test, NUMBER_OF_TEST_EXAMPLES * LA_CHUNKS * sizeof(int));

  read_file();

  cudaMallocManaged((void **)&devStates, GRID_SIZE * BLOCK_SIZE * 
                  sizeof(curandState));

  setup_kernel<<<GRID_SIZE,BLOCK_SIZE>>>(devStates);

  cudaDeviceSynchronize();

  MultiClassTsetlinMachine<2> mc_tm;

  fp = fopen("./statistics.txt","w");
  if (fp == NULL) {
    printf("Error opening\n");
    exit(EXIT_FAILURE);
  }

  for (int e = 0; e < EXPERIMENTS; ++e) {
    printf("\nEXPERIMENT %d\n", e+1);
    mc_tm.initialize();
    for (int i = 0; i < EPOCHS; ++i) {
      printf("\n##### EPOCH %d #####\n", i+1);

      clock_t start, end;
      double gpu_time_testing, gpu_time_training;

      start = clock();
      mc_tm.fit(devStates, X_train, y_train, NUMBER_OF_TRAINING_EXAMPLES, S, 1);
      end = clock();
      gpu_time_training = ((double) (end - start)) / CLOCKS_PER_SEC;

      start = clock();
      mc_tm.evaluate(X_test, y_test, NUMBER_OF_TEST_EXAMPLES);
      end = clock();
      gpu_time_testing = ((double) (end - start)) / CLOCKS_PER_SEC;

      for (int n = 0; n < 2; ++n) {
        printf("\n-- CLASS %d --\n\n", n+1);
              
        float precision = 1.0 * mc_tm.true_positive[n] / (mc_tm.true_positive[n] + mc_tm.false_positive[n]);
        printf("PRECISION: %.3f\n", precision);
        float recall = 1.0 * mc_tm.true_positive[n] / (mc_tm.true_positive[n] + mc_tm.false_negative[n]);
        printf("RECALL: %.3f\n", recall);
        float fscore = 2 * precision * recall / (precision + recall);
        printf("F-SCORE: %.3f\n", fscore);
        
        fprintf(fp, "%d %d %d %d %d %d %d %.4f %.4f %.4f %f %f\n", e, i, n, mc_tm.true_positive[n], mc_tm.false_positive[n],
          mc_tm.true_negative[n], mc_tm.false_negative[n], precision, recall, fscore, gpu_time_training, gpu_time_testing);
        fflush(fp);
      }
      printf("\n");
      printf("TRAINING TIME: %f\n", gpu_time_training);
      printf("TESTING TIME: %f\n", gpu_time_testing);
    }
  }

  fclose(fp);
 
  delete &mc_tm;

  cudaFree(devStates);
  cudaFree(X_train);
  cudaFree(X_test);

  return 0;
}
