# Text Understanding with the Tsetlin Machine
The tools available here introduce the first approach to text categorization that leverages the recently introduced [Tsetlin Machine](https://arxiv.org/pdf/1804.01508.pdf). They contain the CUDA and C source code implemented for the experiments in our [specific paper](https://arxiv.org/pdf/xxxx.pdf) on using the Tsetlin Machine for Text Categorization. Links to the publicly available datasets used in the paper are provided in the code.

## Background
Although deep learning in the form of convolutional neural networks (CNN), recurrent neural networks (RNNs), and Long Short Term Memory (LSTM) recently has provided a leap ahead in text categorization accuracy, this leap has come at the expense of interpretability and computational complexity. 

The Tsetlin automaton developed by M.L. Tsetlin in the Soviet Union in the late 1950s represents a fundamental and versatile learning mechanism; arguably even more so than the artificial neuron. While the Tsetlin Machine is able to form complex non-linear patterns, the propositional formulae it composes have also turned out to be particularly suitable for human interpretation.

In all brevity, the Tsetlin Machine represents the terms of a text as propositional variables. From these, categories are captured using simple propositional formulae, such as:  if “rash” and “reaction” and “penicillin” then Allergy. The Tsetlin Machine learns these formulae from a labelled text, utilizing conjunctive clauses to represent the particular facets of each category.  Indeed, even the absence of terms (negated features) can be used for categorization purposes. The figure below is from our [text categorization paper](https://arxiv.org/xxxx.pdf) and captures the essence of this process (the paper explains the details of the figure). It reflects a running example from the medical domain, where we use the Tsetlin Machine to detect patient allergies in Electronic Health Records. 

![alt text](https://github.com/cair/TextUnderstandingTsetlinMachine/blob/master/Figure4.tif)

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

For compiling the necessary files, use:
nvcc -O3 TsetlinMachineIMDB.cu TsetlinMachine.cu TsetlinMachineKernels.cu -lcurand

## Citation
Please cite the relevant Tsetlin Machine arXiv papers if you use the Tsetlin Machie in your work:

@article{granmo2018tsetlin, 
title={The Tsetlin Machine-A Game Theoretic Bandit Driven Approach to Optimal Pattern Recognition with Propositional Logic}, 
author={Granmo, Ole-Christoffer}, 
journal={arXiv preprint arXiv:1804.01508}, year={2018} 

{Reference to the second paper on using the Tsetlin Machine for Text Categorization is temporarily unavailable until published]. 

## Licence
To be added.

