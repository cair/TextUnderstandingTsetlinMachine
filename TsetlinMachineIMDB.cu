/*

Copyright (c) 2018 Ole-Christoffer Granmo

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

#define NUMBER_OF_EXAMPLES 25000

#define EXPERIMENTS 100
#define EPOCHS 200

#define S 27.0

#define DEVICE 1

int y_train[NUMBER_OF_EXAMPLES], y_test[NUMBER_OF_EXAMPLES];
int *X_train;
int *X_test;

void read_file(void)
{
  FILE * fp;
  char * line = NULL;
  size_t len = 0;

  const char *s = " ";
  char *token = NULL;

  fp = fopen("IMDBTrainingData.txt", "r");
  if (fp == NULL) {
    printf("Error opening\n");
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < NUMBER_OF_EXAMPLES; i++) {
    getline(&line, &len, fp);

    token = strtok(line, s);
    for (int j = 0; j < FEATURES; j++) {
      X_train[i*FEATURES + j] = atoi(token);
      token=strtok(NULL,s);
    }
    y_train[i] = atoi(token);
  }
  fclose(fp);

  fp = fopen("IMDBTestData.txt", "r");
  if (fp == NULL) {
    printf("Error opening\n");
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < NUMBER_OF_EXAMPLES; i++) {
    getline(&line, &len, fp);

    token = strtok(line, s);
    for (int j = 0; j < FEATURES; j++) {
      X_test[i*FEATURES + j] = atoi(token);
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

  cudaMallocManaged(&X_train, NUMBER_OF_EXAMPLES * FEATURES*sizeof(int));
  cudaMallocManaged(&X_test, NUMBER_OF_EXAMPLES * FEATURES*sizeof(int));

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
    fprintf(fp, "\nEXPERIMENT %d\n", e+1);
    mc_tm.initialize();
    for (int i = 0; i < EPOCHS; ++i) {
      printf("\nEPOCH %d\n", i+1);
      fprintf(fp, "\nEPOCH %d\n", i+1);

      mc_tm.fit(devStates, X_train, y_train, NUMBER_OF_EXAMPLES, S, 1);

      mc_tm.evaluate(X_test, y_test, NUMBER_OF_EXAMPLES);

      for (int n = 0; n < 2; ++n) {
        printf("\nCLASS %d\n\n", n+1);
        printf("TRUE POSITIVE: %d\n", mc_tm.true_positive[n]);
        printf("FALSE POSITIVE: %d\n", mc_tm.false_positive[n]);
        printf("TRUE NEGATIVE: %d\n", mc_tm.true_negative[n]);
        printf("FALSE NEGATIVE: %d\n", mc_tm.false_negative[n]);
        printf("\n");
        printf("PRECISION: %.3f\n", 1.0 * mc_tm.true_positive[n] / (mc_tm.true_positive[n] + mc_tm.false_positive[n]));
        printf("RECALL: %.3f\n", 1.0 * mc_tm.true_positive[n] / (mc_tm.true_positive[n] + mc_tm.false_negative[n]));
        printf("ACCURACY: %.3f\n", 1.0 * (mc_tm.true_positive[n] + mc_tm.true_negative[n])/NUMBER_OF_EXAMPLES);

        fprintf(fp, "\nCLASS %d\n\n", n+1);
        fprintf(fp, "TRUE POSITIVE: %d\n", mc_tm.true_positive[n]);
        fprintf(fp, "FALSE POSITIVE: %d\n", mc_tm.false_positive[n]);
        fprintf(fp, "TRUE NEGATIVE: %d\n", mc_tm.true_negative[n]);
        fprintf(fp, "FALSE NEGATIVE: %d\n", mc_tm.false_negative[n]);
        fprintf(fp, "\n");
        fprintf(fp, "PRECISION: %.3f\n", 1.0 * mc_tm.true_positive[n] / (mc_tm.true_positive[n] + mc_tm.false_positive[n]));
        fprintf(fp, "RECALL: %.3f\n", 1.0 * mc_tm.true_positive[n] / (mc_tm.true_positive[n] + mc_tm.false_negative[n]));
        fprintf(fp, "ACCURACY: %.3f\n", 1.0 * (mc_tm.true_positive[n] + mc_tm.true_negative[n])/NUMBER_OF_EXAMPLES);
        fflush(fp);
      }
    }
  }

  fclose(fp);
 
  delete &mc_tm;

  cudaFree(devStates);
  cudaFree(X_train);
  cudaFree(X_test);

  return 0;
}