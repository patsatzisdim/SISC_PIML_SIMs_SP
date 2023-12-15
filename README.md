# Physics Informed Machine Learning for the approximation of Slow Invariant Manifolds of Singularly Perturbed Systems

*If you use or modify for research purposes this software, please cite our paper as below:*

**Patsatzis, D. G., Fabiani, G., Russo, L., & Siettos, C. (2023). Slow Invariant Manifolds of Singularly Perturbed Systems via Physics-Informed Machine Learning. arXiv preprint arXiv:2309.07946.**

*Under review in SIAM Journal of Computational Sciences*

Last updated by Patsatzis D. G., 15 Dec 2023

We present a ``Physics-Informed Machine-Learning`` method for the approximation of ``Slow Invariant Manifolds`` of ``Singularly Perturbed Systems``, providing  functionals in an explicit form that facilitate the construction and numerical integration of reduced order models (ROMs).
We use ``Feedforward Neural Networks`` (FNNs) and ``Random Projection Neural Networks`` (RPNNs) to solve the PDE corresponding to the ``Invariance Equation`` (IE) within the ``Geometric Singular Perturbation Theory`` (GSPT) framework.

The efficiency the scheme is compared with analytic SIM approximation provided by the well-established GSPT methods of IE and ``Computational Singular Perturbation``.

For illustration, we provide three benchmark problems: the Michaelis-Menten (MM), the Target Mediated Drug Disposition (TMDD) reaction mechanism, and the 3D Sel'kov model.

Keywords: Physics-informed machine learning, Slow invariant manifolds, Singular perturbed systems, Random Projection Neural Networks

DISCLAIMER:
This software is provided "as is" without warranty of any kind.

# Software outline

The three benchmark problems (MM, TMDD and 3D Selkov model) are include in folders MM_src, TMDD_src and TLC_src, respenctively.
Each folder contains 3 main routines:
1) createTrainTestSets.m that creates the training/validation and test data sets; see Algorithm SM3.1 in the manuscript
2) PIML_SLFNN.m that that solves the PIML optimization problem using single layer FNNs and allows for symoblic, numerical or automatic differentiation; see Algorithm SM3.2 in the manuscript
3) PIML_RPNN.m that that solves the PIML optimization problem using RPNNs; see Algorithm SM3.2 in the manuscript

Dependancies:

The createTrainTestSets.m main routine depends on get"Problem_name"data.m routine that integrates the given SPS problem (provided as external function) and samples data from trajectories in the desired domain.

The PIML_SLFNN.m and PIML_RPNN.m routines require (i) the training and test data sets (constructed from createTrainTestSets.m) and (ii) the RHS and Jacobian of the given SPS problem (provided as external functions) 

The analytic functionals of the SIM approximations provided by IE and CSP are provided for the given SPS problems; see Appendix in the manuscript.

**Reproducing our results**
For each SPS problem 
(i)   run createTrainTestSets.m to get the training and test data sets (already created as "Problem_name"Train and "Problem_name"Test mat files)
(ii)  run PIML_SLFNN.m and PIML_RPNN.m to train the PIML scheme and evaluate its accuracy (pretrained NNs in learned_PI_SLFNN and learned_PI_RPNN mat files)
(iii) run Plots.m to reproduce the plots in the manuscript (select figure ID)

# Finding SIM approximations with the PIML schem for your own SPS problem

The following is an outline of the steps you would need to follow (for further details see Algorithms 2.1, SM3.1 and SM3.2 in the manuscript)
1) Given your SPS system, set up the RHS and Jacobian external functions 
2) Define the domain $\Omega \times I$ where you want to derive a SIM approximation (in routine createTrainTestSets.m)
3) Create the training/validation and test data sets.
4) Choose the ML structure by selecting PIML_SLFNN.m or PIML_RPNN.m
5) Set hyperparameters for the solution of the PIML scheme 
6) Run many runs in PIML_SLFNN.m or PIML_RPNN.m to select best learned parameters (according to validation set)
7) Produce your own plots for visualization