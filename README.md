# MIPVerify.jl

[![Build Status](https://travis-ci.org/vtjeng/MIPVerify.jl.svg?branch=master)](https://travis-ci.org/vtjeng/MIPVerify.jl) [![codecov.io](http://codecov.io/github/vtjeng/MIPVerify.jl/coverage.svg?branch=master)](http://codecov.io/github/vtjeng/MIPVerify.jl?branch=master)

Recent work has shown that neural networks are vulnerable to _adversarial examples_, with small perturbations to input potentially leading to large changes in the output. In the context of image classification, the perturbed input is often indistinguishable from the original input, but can lead to misclassifications into any target category chosen by the adversary.

Finding _some_ adversarial example can be done efficiently via iterative searches which seek to optimize over a selected loss function. However, these approaches do not guarantee that they will find an adversarial example if one exists, and cannot provide any guarantees about the distance to the _closest_ adversarial example.

`MIPVerify.jl` enables users to find the closest adversarial example for networks that are piecewise affine (for example, deep networks with ReLU and maxpool units). The optimization problem is translated into a mixed-integer programming (MIP) model, which can then be solved by any solver supported by [JuMP](https://github.com/JuliaOpt/JuMP.jl).

The package provides
  + High-level abstractions for common types of neural network layers (convolution layers, fully connected layers, and pooling layers).
  + Tight specification of non-linear constraints between decision variables, enabling efficient solves.
  + Support for restricting allowable perturbations to a specified families of perturbations [1].

In addition, we provide utility functions to work with the MNIST dataset.

See the companion paper for more details and results:

**Verifying Neural Networks with Mixed Integer Programming**
_Vincent Tjeng, Russ Tedrake_
https://arxiv.org/abs/1711.07356

See the latest stable documentation for a list of features, installation instructions, and a quick-start guide. Installation should only take a couple of minutes, including installing Julia itself. See the notebooks directory for some usage examples.

## Citing this library
```
@article{tjeng2017verifying,
  title={Verifying Neural Networks with Mixed Integer Programming},
  author={Tjeng, Vincent and Tedrake, Russ},
  journal={arXiv preprint arXiv:1711.07356},
  year={2017}
}
```

[1] Currently, we have support for _additive_ perturbations, where every pixel can be modified independently, and _blurring_ perturbations, where the image is convolved with an adversarial blurrign kernel.