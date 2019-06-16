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

This code implements the Tsetlin Machine from paper arXiv:1804.01508
https://arxiv.org/abs/1804.01508

*/

#include <curand.h>
#include <curand_kernel.h>

#define INT_SIZE 32

#define LA_CHUNKS (((2*FEATURES-1)/INT_SIZE + 1))
#define CLAUSE_CHUNKS ((CLAUSES-1)/INT_SIZE + 1)

#if ((FEATURES*2) % 32 != 0)
#define FILTER (~(0xffffffff << ((FEATURES*2) % INT_SIZE)))
#else
#define FILTER 0xffffffff
#endif

class TsetlinMachine { 
	/* Tsetlin Machine data structures */

	int *class_sum;

	int *ta_state;
	int *clause_output;
	int *clause_feedback;
	int *all_exclude;

public:
	TsetlinMachine();

	~TsetlinMachine();

	void update(curandState *devStates, int *Xi, int target, float s);

	int score(int *Xi);

	int get_state(int id);

	void initialize();
};






