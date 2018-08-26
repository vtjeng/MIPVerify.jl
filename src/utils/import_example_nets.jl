"""
$(SIGNATURES)

Makes named example neural networks available as a [`NeuralNet`](@ref) object.

# Arguments
* `name::String`: Name of example neural network. Options:
    * `'MNIST.n1'`: 
        * Architecture: Two fully connected layers with 40 and 20 units.
        * Training: Trained regularly with no attempt to increase robustness.
    * `'MNIST.WK17a_linf0.1_authors'`. 
        * Architecture: Two convolutional layers (stride length 2) with 16 and 
          32 filters respectively (size 4 × 4 in both layers), followed by a 
          fully-connected layer with 100 units.
        * Training: Network trained to be robust to attacks with \$l_\\infty\$ norm
          at most 0.1 via method in [Provable defenses against adversarial examples 
          via the convex outer adversarial polytope](https://arxiv.org/abs/1711.00851). 
          Is MNIST network for which results are reported in that paper.
    * `'MNIST.RSL18a_linf0.1_authors'`. 
        * Architecture: One fully connected layer with 500 units. 
        * Training: Network trained to be robust to attacks with \$l_\\infty\$ norm
          at most 0.1 via method in [Certified Defenses against Adversarial Examples
          ](https://arxiv.org/abs/1801.09344). 
          Is MNIST network for which results are reported in that paper.
"""
function get_example_network_params(name::String)::NeuralNet
    if name == "MNIST.n1"       
        param_dict = prep_data_file(joinpath("weights", "mnist"), "n1.mat") |> matread
        fc1 = get_matrix_params(param_dict, "fc1", (784, 40))
        fc2 = get_matrix_params(param_dict, "fc2", (40, 20))
        logits = get_matrix_params(param_dict, "logits", (20, 10))
        
        nn = Sequential([
            Flatten(4),
            fc1, ReLU(interval_arithmetic),
            fc2, ReLU(),
            logits], name)
        return nn
    elseif name == "MNIST.WK17a_linf0.1_authors"
        param_dict = prep_data_file(joinpath("weights", "mnist", "WK17a", "linf0.1"), "master_seed_1_epochs_100.mat") |> matread
        conv1 = get_conv_params(param_dict, "conv1", (4, 4, 1, 16), expected_stride = 2)
        conv2 = get_conv_params(param_dict, "conv2", (4, 4, 16, 32), expected_stride = 2)
        fc1 = get_matrix_params(param_dict, "fc1", (1568, 100))
        logits = get_matrix_params(param_dict, "logits", (100, 10))
        
        nn = Sequential([
            conv1, ReLU(interval_arithmetic),
            conv2, ReLU(),
            Flatten([1, 3, 2, 4]),
            fc1, ReLU(),
            logits], name)
        return nn
    elseif name == "MNIST.RSL18a_linf0.1_authors"
        param_dict = prep_data_file(joinpath("weights", "mnist", "RSL18a", "linf0.1"), "two-layer.mat") |> matread
        fc1 = get_matrix_params(param_dict, "fc1", (784, 500))
        logits = get_matrix_params(param_dict, "logits", (500, 10))
        
        nn = Sequential([
            Flatten([1, 3, 2, 4]),
            fc1, ReLU(interval_arithmetic),
            logits], name)
        return nn
    else
        throw(DomainError("No example network named $name."))
    end
end

# TODO (vtjeng): Add mnist networks Ragunathan/Steinhardt/Liang.
# TODO (vtjeng): Make network naming case insensitive.