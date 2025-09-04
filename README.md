# Quantifying policy uncertainty in generative flow networks with uncertain rewards via surrogate modelling

This repository provides code supporting research into uncertainty quantification (UQ) for generative flow networks (GFlowNets).

We focus our numerical experiments on four example problems

1. A discrete grid-world
2. A continuous grid-world
3. A symbolic regression task
4. A Bayesian structure learning task

For Examples 1-3, we provide

1. A script of helper functions
2. A training script for an ensemble of models
3. Two ensembles of pretrained GFlowNets, one for training and one for testing
4. A notebook executing the surrogate modelling with polynomial chaos expansions (PCE) and multilayer perceptrons (MLP)

For the Bayesian structure learning task, the training code is taken from Deleu et al. (2022) Bayesian Structure Learning with Generative Flow Networks, from their repository:
https://github.com/tristandeleu/jax-dag-gflownet

We do not provide the ensembles of models for this task, as the files are prohibitively large, but we provide the training scripts needed to train them yourself.

The pyproject.toml contains the necessary Python libraries and versions needed to run this code, with the exception that JAX is needed to train the models from Deleu et al.
