# MIPVerify.jl

[![Build Status](https://travis-ci.org/vtjeng/MIPVerify.jl.svg?branch=master)](https://travis-ci.org/vtjeng/MIPVerify.jl)
[![codecov.io](http://codecov.io/github/vtjeng/MIPVerify.jl/coverage.svg?branch=master)](http://codecov.io/github/vtjeng/MIPVerify.jl?branch=master)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://vtjeng.github.io/MIPVerify.jl/stable)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://vtjeng.github.io/MIPVerify.jl/latest)

_A package for evaluating the robustness of neural networks using Mixed Integer Programming (MIP). See the [companion paper](https://arxiv.org/abs/1711.07356) for full details and results._

**Evaluating Robustness of Neural Networks with Mixed Integer Programming**
_Vincent Tjeng, Kai Xiao, Russ Tedrake_
https://arxiv.org/abs/1711.07356

## Getting Started
See the [documentation](https://vtjeng.github.io/MIPVerify.jl/latest) for [installation instructions](https://vtjeng.github.io/MIPVerify.jl/latest/#Installation-1), a [quick-start guide](https://nbviewer.jupyter.org/github/vtjeng/MIPVerify.jl/blob/master/examples/00_quickstart.ipynb), and [additional examples](https://nbviewer.jupyter.org/github/vtjeng/MIPVerify.jl/tree/master/examples/). Installation should only take a couple of minutes, including installing Julia itself.

## Why Verify Neural Networks?
Neural networks trained only to optimize for training accuracy have been shown to be vulnerable to _adversarial examples_, with small perturbations to input potentially leading to large changes in the output. In the context of image classification, the perturbed input is often indistinguishable from the original input, but can lead to misclassifications into any target category chosen by the adversary.

There is now a large body of work proposing defense methods to produce classifiers that are more robust to adversarial examples. However, as long as a defense is evaluated only via attacks that find local optima, we have no guarantee that the defense actually increases the robustness of the classifier produced.

Fortunately, we _can_ evaluate robustness to adversarial examples in a principled fashion. One option is to determine (for each test input) the minimum distance to the closest adversarial example, which we call the _minimum adversarial distortion_. The second option is to determine the _adversarial test accuracy_, which is the proportion of the test set for which no bounded perturbation causes a misclassification. An increase in the mean minimum adversarial distortion or in the adversarial test accuracy indicates an improvement in robustness.

Determining the minimum adversarial distortion for some input (or proving that no bounded perturbation of that input causes a misclassification) corresponds to solving an optimization problem. For piecewise-linear neural networks, the optimization problem can be expressed as a mixed-integer linear programming (MILP) problem.

## Features
`MIPVerify.jl` translates your query on the robustness of a neural network for some input into an MILP problem, which can then be solved by any solver supported by [JuMP](https://github.com/JuliaOpt/JuMP.jl). Efficient solves are enabled by tight specification of ReLU and maximum constraints and a progressive bounds tightening approach where time is spent refining bounds only if doing so could provide additional information to improve the problem formulation.

The package provides
  + High-level abstractions for common types of neural network layers:
    + Layers that are linear transformations (fully-connected, convolution, and average-pooling layers)
    + Layers that use piecewise-linear functions (ReLU and maximum-pooling layers)
  + Support for bounding perturbations to:
    + Perturbations of bounded l-infty norm
    + Perturbations where the image is convolved with an adversarial blurring kernel
  + Utility functions for:
    + Evaluating the robustness of a network on multiple samples in a dataset, with good support for pausing and resuming evaluation or running solvers with different parameters
  + MNIST and CIFAR10 datasets for verification
  + Sample neural networks, including the networks verified in our paper.

## Results in Brief
Below is a modified version of Table 1 from our paper, where we report the adversarial error for classifiers to bounded perturbations with l-infinity norm-bound `eps`. For our verifier, a time limit of 120s per sample is imposed. Gaps between our bounds correspond to cases where the solver reached the time limit for some samples. Error is over the full MNIST test set of 10,000 samples.

| Dataset | Training Approach | `eps` | Lower<br>Bound<br>(PGD Error) | Lower<br>Bound<br>(ours) | Upper<br>Bound<br>(SOA)\^ | Upper<br>Bound<br>(ours)| Name in package\* |
|---|---|---|---|---|---|---|---|
| MNIST | [Wong et al. (2017)](https://arxiv.org/abs/1711.00851) | 0.1 |  4.11% | **4.38%** | 5.82% | **4.38%** | `MNIST.WK17a_linf0.1_authors` |
| MNIST | [Ragunathan et al. (2018)](https://arxiv.org/abs/1801.09344) | 0.1 | 11.51% | **14.36%** | 34.77% | **30.81%** | `MNIST.RSL18a_linf0.1_authors` |

\^ Values in this column represent previous state-of-the-art (SOA), as described in our paper.<br>
\* Neural network available for import via listed name using `get_example_network_params`.

## Citing this Library
```
@article{tjeng2017evaluating,
  title={Evaluating Robustness of Neural Networks with Mixed Integer Programming},
  author={Tjeng, Vincent and Xiao, Kai and Tedrake, Russ},
  journal={arXiv preprint arXiv:1711.07356},
  year={2017}
}
```
