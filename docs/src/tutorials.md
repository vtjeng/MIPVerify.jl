# Tutorials
We suggest getting started with the tutorials.

## Quickstart
[A basic demonstration](https://nbviewer.jupyter.org/github/vtjeng/MIPVerify.jl/blob/master/examples/00_quickstart.ipynb) on how to find adversarial examples for a pre-trained example network on the MNIST dataset.

## Importing your own neural net
[Explains](https://nbviewer.jupyter.org/github/vtjeng/MIPVerify.jl/blob/master/examples/01_importing_your_own_neural_net.ipynb) how to import your own network for verification.

## Finding adversarial examples, in depth
[Discusses](https://nbviewer.jupyter.org/github/vtjeng/MIPVerify.jl/blob/master/examples/02_finding_adversarial_examples_in_depth.ipynb) the various parameters you can select for `find_adversarial_example`. We explain how to

  + Better specify targeted labels for the perturbed image (including multiple targeted labels)
  + Have more precise control over the activations in the output layer
  + Restrict the family of perturbations (for example to the blurring perturbations discussed in our paper)
  + Select whether you want to minimize the $L_1$, $L_2$ or $L_\infty$ norm of the perturbation.
  + Determine whether you are rebuilding the model expressing the constraints of the neural network from scratch, or loading the model from cache.
  + Modify the amount of time dedicated to building the model (by selecting the `tightening_algorithm`, and/or passing in a custom `tightening_solver`).

For Gurobi, we show how to specify solver settings to:
  + Mute output
  + Terminate early if:
    + A time limit is reached
    + Lower bounds on robustness are proved (that is, we prove that no adversarial example can exist closer than some threshold)
    + An adversarial example is found that is closer to the input than expected
    + The gap between the upper and lower objective bounds falls below a selected threshold

## Interpreting the output of `find_adversarial_example`
[Walks you through](https://nbviewer.jupyter.org/github/vtjeng/MIPVerify.jl/blob/master/examples/03_interpreting_the_output_of_find_adversarial_example.ipynb) the output dictionary produced by a call to `find_adversarial_example`.

## Managing log output
[Explains how](https://nbviewer.jupyter.org/github/vtjeng/MIPVerify.jl/blob/master/examples/04_managing_log_output.ipynb) to get more granular log settings and to write log output to file.