# Text Understanding with the Tsetlin Machine
This project contains the source code for our text categorization approach based on the recently introduced [Tsetlin Machine](https://arxiv.org/pdf/1804.01508.pdf). We have included the CUDA implementation, and will later also include the C source code. This is the code that we used to produce the empirical results in our paper on using the Tsetlin Machine for text categorization (https://arxiv.org/abs/1809.04547). Links to the publicly available datasets used in the paper are provided in the code.

## Background
Although deep learning in the form of convolutional neural networks (CNN), recurrent neural networks (RNNs), and Long Short Term Memory (LSTM) recently has provided a leap ahead in text categorization accuracy, this leap has come at the expense of interpretability and computational complexity. 

The Tsetlin automaton developed by M.L. Tsetlin in the Soviet Union in the late 1950s represents a fundamental and versatile learning mechanism; arguably even more so than the artificial neuron. While the Tsetlin Machine is able to form complex non-linear patterns, the propositional formulae it composes have also turned out to be particularly suitable for human interpretation.

In all brevity, the Tsetlin Machine represents the terms of a text as propositional variables. From these, categories are captured using simple propositional formulae, such as:  if “rash” and “reaction” and “penicillin” then Allergy. The Tsetlin Machine learns these formulae from a labelled text, utilizing conjunctive clauses to represent the particular facets of each category.  Indeed, even the absence of terms (negated features) can be used for categorization purposes. The figure below is from our [text categorization paper](https://arxiv.org/xxxx.pdf) and captures the essence of this process (the paper explains the details of the figure). It reflects a running example from the medical domain, where we use the Tsetlin Machine to detect patient allergies in Electronic Health Records. 

![Figure 4](https://raw.githubusercontent.com/bluebyte9001/TextUnderstandingTsetlinMachine/master/Figure4.png)

The combination of its computational simplicity, accuracy, and finally results that are highly interpretable, leaves the Tsetlin Machine worthy of further exploration in many text analysis directions.  

We refer to the original [seed paper](https://arxiv.org/pdf/1804.01508.pdf) on the Tsetlin Machine, and our [specific paper](https://arxiv.org/pdf/xxxx.pdf) on using the Tsetlin Machine for text categorization, for details on how the method works.

Also notice that research on the Tsetlin Machine for text understanding is an on-going project at [the Centre for Artificial Intelligence Research CAIR](https://cair.uia.no/) at the University of Agder, Norway. More information will be added as the project advances.

## Requirements
•	Python 2.7.x https://www.python.org/downloads/

•	Numpy http://www.numpy.org/

•	Cython http://cython.org/

•	CUDA https://developer.nvidia.com/cuda

•	NLTK http://www.nltk.org

•	Scikit-learn http://scikit-learn.org

•	Gcc http://gcc.gnu.org


## Instructions for use

For compiling the necessary files, use the following command:
nvcc -O3 TsetlinMachineIMDB.cu TsetlinMachine.cu TsetlinMachineKernels.cu -lcurand

## Citation
Please cite the relevant Tsetlin Machine arXiv papers if you use the Tsetlin Machie in your work:

@article{granmo2018tsetlin, 
title={The Tsetlin Machine-A Game Theoretic Bandit Driven Approach to Optimal Pattern Recognition with Propositional Logic}, 
author={Granmo, Ole-Christoffer}, 
journal={arXiv preprint arXiv:1804.01508}, year={2018} 

{Reference to the second paper on using the Tsetlin Machine for Text Categorization is temporarily unavailable until published]. 

## Licence
Copyright (c) 2018 Geir Thore Berge and Ole-Christoffer Granmo

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
