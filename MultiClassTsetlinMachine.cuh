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

#include "TsetlinMachine.cuh"

static void shuffle(int *array, size_t n)
{
	if (n > 1) {
		size_t i;
		for (i = 0; i < n - 1; i++) {
			size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
		    int t = array[j];
		    array[j] = array[i];
		    array[i] = t;
		}
    }
}

template<int N> class MultiClassTsetlinMachine {
	TsetlinMachine tsetlin_machines[N];

public:
	int false_positive[N], false_negative[N], true_positive[N], true_negative[N];

	MultiClassTsetlinMachine(void)
	{
	}

	~MultiClassTsetlinMachine(void)
	{
		for (int i = 0; i < N; i++) {
			delete &tsetlin_machines[i];
		}		
	}

	void initialize() 
	{
		for (int tm = 0; tm < N; tm++) {
			tsetlin_machines[tm].initialize();
		}
	}

	void fit(curandState *devStates, int *X, int *y, int number_of_examples, float s, int epochs)
	{
		clock_t start_total, end_total;
	  	double gpu_time_used;

		int *index = (int *)malloc(sizeof(int)*number_of_examples);

		for (int i = 0; i < number_of_examples; i++) {
	    	index[i] = i;
	  	}

	  	for (int epoch = 0; epoch < epochs; epoch++) {
	  		shuffle(index, number_of_examples);

	  		start_total = clock();

		    for (int i = 0; i < number_of_examples; i++) {
		    	update(devStates, &X[index[i]*FEATURES], y[index[i]], s);
		    }
		
			end_total = clock();

			gpu_time_used = ((double) (end_total - start_total)) / CLOCKS_PER_SEC;

		    printf("\nTIME: %f\n", gpu_time_used);
		}

		free(index);
	}

	void update(curandState *devStates, int *Xi, int y, float s)
	{
		tsetlin_machines[y].update(devStates, Xi, 1, s);

		int neg_y = (int)N * 1.0*rand()/RAND_MAX;
		while (neg_y == y) {
			neg_y = (int)N * 1.0*rand()/RAND_MAX;
		}

		tsetlin_machines[neg_y].update(devStates, Xi, 0, s);
	}

	void evaluate(int *X, int *y, int number_of_examples)
	{
		clock_t start_total, end_total;
	  	double gpu_time_used;

		start_total = clock();

		for (int n = 0; n < N; n++) {
		    true_negative[n] = 0;
		    true_positive[n] = 0;
		    false_negative[n] = 0;
		    false_positive[n] = 0;
		}

	    for (int i = 0; i < number_of_examples; i++) {
	    	int max_category = 0;
	    	int max_score = tsetlin_machines[0].score(&X[i*FEATURES]);
	    	for (int tm = 1; tm < N; tm++) {
	      		int score = tsetlin_machines[tm].score(&X[i*FEATURES]);
	      		if (score > max_score) {
	      			max_score = score;
	      			max_category = tm;
	      		}
	      	}

	      	if (max_category == y[i]) {
	      		true_positive[y[i]]++;

	      		for (int n = 0; n < N; n++) {
	      			if (n != max_category) {
	      				true_negative[n]++;
	      			}
	      		}
	      	}

	      	if (max_category != y[i]) {
	      		false_negative[y[i]]++;
	      		false_positive[max_category]++;
	      		for (int n = 0; n < N; n++) {
	      			if (n != max_category && n != y[i]) {
	      				true_negative[n]++;
	      			}
	      		}
	      	}
	    }

	    end_total = clock();
	    gpu_time_used = ((double) (end_total - start_total)) / CLOCKS_PER_SEC;
	    printf("TOTAL EVALUATION TIME: %f\n", gpu_time_used);
	}

	int predict(int *Xi)
	{
		int max_category = 0;
	    int max_score = tsetlin_machines[0].score(Xi);

	    for (int i = 1; i < N; i++) {
	  		int score = tsetlin_machines[i].score(Xi);
	  		if (score > max_score) {
	  			max_score = score;
	  			max_category = i;
	  		}
	    }

		return max_category;
	}
	
};