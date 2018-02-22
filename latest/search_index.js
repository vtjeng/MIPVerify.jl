var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#MIPVerify-1",
    "page": "Home",
    "title": "MIPVerify",
    "category": "section",
    "text": "MIPVerify.jl enables users to verify neural networks that are piecewise affine by finding the closest adversarial example to a selected input."
},

{
    "location": "index.html#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": ""
},

{
    "location": "index.html#Installing-Julia-1",
    "page": "Home",
    "title": "Installing Julia",
    "category": "section",
    "text": "Download links and more detailed instructions are available on the Julia website. The latest release of this package requires version 0.6 of Julia.warning: Warning\nDo not use apt-get or brew to install Julia, as the versions provided by these package managers tend to be out of date."
},

{
    "location": "index.html#Installing-MIPVerify-1",
    "page": "Home",
    "title": "Installing MIPVerify",
    "category": "section",
    "text": "Once you have Julia installed, install the latest tagged release of MIPVerify by runningPkg.add(\"MIPVerify\")"
},

{
    "location": "index.html#Getting-Started-1",
    "page": "Home",
    "title": "Getting Started",
    "category": "section",
    "text": "The best way to get started is to follow our quickstart tutorial, which demonstrates how to find adversarial examples for a pre-trained example network on the MNIST dataset. Once you\'re done with that, you can explore our other tutorials depending on your needs."
},

{
    "location": "tutorials.html#",
    "page": "Tutorials",
    "title": "Tutorials",
    "category": "page",
    "text": ""
},

{
    "location": "tutorials.html#Tutorials-1",
    "page": "Tutorials",
    "title": "Tutorials",
    "category": "section",
    "text": ""
},

{
    "location": "tutorials.html#Quickstart-1",
    "page": "Tutorials",
    "title": "Quickstart",
    "category": "section",
    "text": "A basic demonstration on how to find adversarial examples for a pre-trained example network on the MNIST dataset."
},

{
    "location": "tutorials.html#Importing-your-own-neural-net-1",
    "page": "Tutorials",
    "title": "Importing your own neural net",
    "category": "section",
    "text": "Explains how to import your own network for verification."
},

{
    "location": "tutorials.html#Finding-adversarial-examples,-in-depth-1",
    "page": "Tutorials",
    "title": "Finding adversarial examples, in depth",
    "category": "section",
    "text": "Discusses the various parameters you can select for find_adversarial_example. We explain how toBetter specify targeted labels for the perturbed image (including multiple targeted labels)\nHave more precise control over the activations in the output layer\nRestrict the family of perturbations (for example to the blurring perturbations discussed in our paper)\nSelect whether you want to minimize the L_1, L_2 or L_infty norm of the perturbation.\nDetermine whether you are rebuilding the model expressing the constraints of the neural network from scratch, or loading the model from cache.\nModify the amount of time dedicated to building the model (by passing in a custom model_build_solver).For Gurobi, we show how to specify solver settings to:Mute output\nTerminate early if:\nA time limit is reached\nLower bounds on robustness are proved (that is, we prove that no adversarial example can exist closer than some threshold)\nAn adversarial example is found that is closer to the input than expected\nThe gap between the upper and lower objective bounds falls below a selected threshold"
},

{
    "location": "tutorials.html#Interpreting-the-output-of-find_adversarial_example-1",
    "page": "Tutorials",
    "title": "Interpreting the output of find_adversarial_example",
    "category": "section",
    "text": "Walks you through the output dictionary produced by a call to find_adversarial_example."
},

{
    "location": "tutorials.html#Managing-log-output-1",
    "page": "Tutorials",
    "title": "Managing log output",
    "category": "section",
    "text": "Explains how getting more granular log settings and writing log output to file."
},

{
    "location": "finding_adversarial_examples.html#",
    "page": "Finding Adversarial Examples",
    "title": "Finding Adversarial Examples",
    "category": "page",
    "text": ""
},

{
    "location": "finding_adversarial_examples.html#Finding-Adversarial-Examples-1",
    "page": "Finding Adversarial Examples",
    "title": "Finding Adversarial Examples",
    "category": "section",
    "text": "find_adversarial_example is the core function that you will be calling to  find adversarial examples. To avoid spending time verifying the wrong network, we suggest that you check that the network gets reasonable performance on the test set using [frac_correct]."
},

{
    "location": "finding_adversarial_examples.html#Index-1",
    "page": "Finding Adversarial Examples",
    "title": "Index",
    "category": "section",
    "text": "Pages   = [\"finding_adversarial_examples.md\"]\nOrder   = [:function, :type]"
},

{
    "location": "finding_adversarial_examples.html#MIPVerify.find_adversarial_example-Tuple{MIPVerify.NeuralNetParameters,Array{#s6,N} where N where #s6<:Real,Union{Array{#s5,1} where #s5<:Integer, Integer},MathProgBase.SolverInterface.AbstractMathProgSolver}",
    "page": "Finding Adversarial Examples",
    "title": "MIPVerify.find_adversarial_example",
    "category": "Method",
    "text": "find_adversarial_example(nnparams, input, target_selection, main_solver; pp, norm_order, tolerance, rebuild, invert_target_selection, model_build_solver)\n\n\nFinds the perturbed image closest to input such that the network described by nnparams classifies the perturbed image in one of the categories identified by the  indexes in target_selection.\n\nmain_solver specifies the solver used.\n\nFormal Definition: If there are a total of n categories, the output vector y has  length n. We guarantee that y[j] - y[i] ≥ tolerance for some j ∈ target_selection  and for all i ∉ target_selection.\n\nNamed Arguments:\n\npp::PerturbationParameters: Defaults to AdditivePerturbationParameters(). Determines   the family of perturbations over which we are searching for adversarial examples.\nnorm_order::Real: Defaults to 1. Determines the distance norm used to determine the    distance from the perturbed image to the original. Supported options are 1, Inf    and 2 (if the main_solver used can solve MIQPs.)\ntolerance: Defaults to 0.0. As above.\nrebuild: Defaults to false. If true, rebuilds model by determining upper and lower   bounds on input to each non-linear unit even if a cached model exists.\ninvert_target_selection: defaults to false. If true, sets target_selection to    be its complement.\nmodel_build_solver: Used to determine the upper and lower bounds on input to each    non-linear unit. Defaults to the same type of solver as the main_solver, with a   time limit of 20s per solver and output suppressed. \n\n\n\n"
},

{
    "location": "finding_adversarial_examples.html#MIPVerify.frac_correct-Tuple{MIPVerify.NeuralNetParameters,MIPVerify.ImageDataset,Int64}",
    "page": "Finding Adversarial Examples",
    "title": "MIPVerify.frac_correct",
    "category": "Method",
    "text": "frac_correct(nnparams, dataset, num_samples)\n\n\nReturns the fraction of items the neural network correctly classifies of the first num_samples of the provided dataset. If there are fewer than num_samples items, we use all of the available samples.\n\nNamed Arguments:\n\nnnparams::NeuralNetParameters: The parameters of the neural network.\ndataset::ImageDataset:\nnum_samples::Int: Number of samples to use.\n\n\n\n"
},

{
    "location": "finding_adversarial_examples.html#Public-Interface-1",
    "page": "Finding Adversarial Examples",
    "title": "Public Interface",
    "category": "section",
    "text": "Modules = [MIPVerify]\nOrder   = [:function, :type]\nPages   = [\"MIPVerify.jl\"]\nPrivate = false"
},

{
    "location": "net_components/overview.html#",
    "page": "Overview",
    "title": "Overview",
    "category": "page",
    "text": ""
},

{
    "location": "net_components/overview.html#Overview-1",
    "page": "Overview",
    "title": "Overview",
    "category": "section",
    "text": "A neural net consists of multiple layers, each of which (potentially) operates on input differently. We represent these objects with NeuralNetParameters and LayerParameters."
},

{
    "location": "net_components/overview.html#Index-1",
    "page": "Overview",
    "title": "Index",
    "category": "section",
    "text": "Pages   = [\"overview.md\"]\nOrder   = [:function, :type]"
},

{
    "location": "net_components/overview.html#MIPVerify.LayerParameters",
    "page": "Overview",
    "title": "MIPVerify.LayerParameters",
    "category": "Type",
    "text": "abstract LayerParameters\n\nSupertype for all types storing the parameters of each layer. Inherit from this to specify your own custom type of layer. Each implementation is expected to:\n\nImplement a callable specifying the output when any input of type JuMPReal is provided.\n\n\n\n"
},

{
    "location": "net_components/overview.html#MIPVerify.StackableLayerParameters",
    "page": "Overview",
    "title": "MIPVerify.StackableLayerParameters",
    "category": "Type",
    "text": "abstract StackableLayerParameters <: MIPVerify.LayerParameters\n\nSupertype for all LayerParameters that can be logically applied in sequence.\n\nAn array of StackableLayerParameters is interpreted as that array of layer being applied to the input sequentially, starting from the leftmost layer. (In functional programming terms, this can be thought of as a sort of fold).\n\n\n\n"
},

{
    "location": "net_components/overview.html#MIPVerify.NeuralNetParameters",
    "page": "Overview",
    "title": "MIPVerify.NeuralNetParameters",
    "category": "Type",
    "text": "abstract NeuralNetParameters\n\nSupertype for all types storing the parameters of a neural net. Inherit from this to specify your own custom architecture of LayerParameters. Each implementation is expected to:\n\nImplement a callable specifying the output when any input of type JuMPReal is provided\nHave a UUID field for the name of the neural network.\n\n\n\n"
},

{
    "location": "net_components/overview.html#MIPVerify.JuMPReal",
    "page": "Overview",
    "title": "MIPVerify.JuMPReal",
    "category": "Constant",
    "text": "\n\n"
},

{
    "location": "net_components/overview.html#PublicInterface-1",
    "page": "Overview",
    "title": "PublicInterface",
    "category": "section",
    "text": "LayerParameters\nStackableLayerParameters\nNeuralNetParameters\nMIPVerify.JuMPReal"
},

{
    "location": "net_components/layers.html#",
    "page": "Layers",
    "title": "Layers",
    "category": "page",
    "text": ""
},

{
    "location": "net_components/layers.html#Layers-1",
    "page": "Layers",
    "title": "Layers",
    "category": "section",
    "text": "Each layer in the neural net corresponds to a struct that simultaneously specifies: 1) the operation being carried out in the layer (recorded in the _type_ of the struct) and 2) the parameters for the operation (recorded in the values of the fields of the struct).When we pass an input array of real numbers to a layer struct, we get an output array of real numbers that is the result of the layer operating on the input.Conversely, when we pass an input array of JuMP variables, we get an output array of JuMP variables, with the appropriate mixed-integer constraints (as determined by the layer) imposed between the input and output."
},

{
    "location": "net_components/layers.html#Index-1",
    "page": "Layers",
    "title": "Index",
    "category": "section",
    "text": "Pages   = [\"layers.md\"]\nOrder   = [:function, :type]"
},

{
    "location": "net_components/layers.html#MIPVerify.Conv2DParameters",
    "page": "Layers",
    "title": "MIPVerify.Conv2DParameters",
    "category": "Type",
    "text": "struct Conv2DParameters{T<:Union{JuMP.AbstractJuMPScalar, Real}, U<:Union{JuMP.AbstractJuMPScalar, Real}} <: MIPVerify.LayerParameters\n\nStores parameters for a 2-D convolution operation.\n\np(x) is shorthand for conv2d(x, p) when p is an instance of Conv2DParameters.\n\nFields:\n\nfilter\nbias\n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.Conv2DParameters-Union{Tuple{Array{T,4}}, Tuple{T}} where T<:Union{JuMP.AbstractJuMPScalar, Real}",
    "page": "Layers",
    "title": "MIPVerify.Conv2DParameters",
    "category": "Method",
    "text": "Conv2DParameters(filter)\n\n\nConvenience function to create a Conv2DParameters struct with the specified filter and zero bias.\n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.ConvolutionLayerParameters",
    "page": "Layers",
    "title": "MIPVerify.ConvolutionLayerParameters",
    "category": "Type",
    "text": "struct ConvolutionLayerParameters{T<:Real, U<:Real} <: MIPVerify.StackableLayerParameters\n\nStores parameters for a convolution layer consisting of a convolution, followed by max-pooling, and a ReLU activation function.\n\np(x) is shorthand for convolution_layer(x, p) when p is an instance of ConvolutionLayerParameters.\n\nFields:\n\nconv2dparams\nmaxpoolparams\n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.ConvolutionLayerParameters-Union{Tuple{Array{T,4},Array{U,1},NTuple{4,Int64}}, Tuple{T}, Tuple{U}} where U<:Real where T<:Real",
    "page": "Layers",
    "title": "MIPVerify.ConvolutionLayerParameters",
    "category": "Method",
    "text": "ConvolutionLayerParameters(filter, bias, strides)\n\n\nConvenience function to create a ConvolutionLayerParameters struct with the specified filter, bias, and strides for the max-pooling operation. \n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.FullyConnectedLayerParameters",
    "page": "Layers",
    "title": "MIPVerify.FullyConnectedLayerParameters",
    "category": "Type",
    "text": "struct FullyConnectedLayerParameters{T<:Real, U<:Real} <: MIPVerify.StackableLayerParameters\n\nStores parameters for a fully connected layer consisting of a matrix multiplication  followed by a ReLU activation function.\n\np(x) is shorthand for fully_connected_layer(x, p) when p is an instance of FullyConnectedLayerParameters.\n\nFields:\n\nmmparams\n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.MaskedFullyConnectedLayerParameters",
    "page": "Layers",
    "title": "MIPVerify.MaskedFullyConnectedLayerParameters",
    "category": "Type",
    "text": "struct MaskedFullyConnectedLayerParameters{T<:Real, U<:Real, V<:Real} <: MIPVerify.StackableLayerParameters\n\nStores parameters for a fully connected layer consisting of a matrix multiplication  followed by a ReLU activation function that is activated selectively depending on the  corresponding value of the mask.\n\np(x) is shorthand for masked_fully_connected_layer(x, p) when p is an instance of MaskedFullyConnectedLayerParameters.\n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.MatrixMultiplicationParameters",
    "page": "Layers",
    "title": "MIPVerify.MatrixMultiplicationParameters",
    "category": "Type",
    "text": "struct MatrixMultiplicationParameters{T<:Real, U<:Real} <: MIPVerify.LayerParameters\n\nStores parameters for a layer that does a simple matrix multiplication.\n\np(x) is shorthand for matmul(x, p) when p is an instance of MatrixMultiplicationParameters.\n\nFields:\n\nmatrix\nbias\n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.MaxPoolParameters-Union{Tuple{N}, Tuple{Tuple{Vararg{Int64,N}}}} where N",
    "page": "Layers",
    "title": "MIPVerify.MaxPoolParameters",
    "category": "Method",
    "text": "MaxPoolParameters(strides)\n\n\nConvenience function to create a PoolParameters struct for max-pooling.\n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.PoolParameters",
    "page": "Layers",
    "title": "MIPVerify.PoolParameters",
    "category": "Type",
    "text": "struct PoolParameters{N} <: MIPVerify.LayerParameters\n\nStores parameters for a pooling operation.\n\np(x) is shorthand for pool(x, p) when p is an instance of PoolParameters.\n\nFields:\n\nstrides\npooling_function\n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.SoftmaxParameters",
    "page": "Layers",
    "title": "MIPVerify.SoftmaxParameters",
    "category": "Type",
    "text": "struct SoftmaxParameters{T<:Real, U<:Real} <: MIPVerify.LayerParameters\n\nStores parameters for a softmax layer consisting of a matrix multiplication with _no_ activation function.\n\nThis simply wraps MatrixMultiplicationParameters to ensure that it is distinguishable as a softmax layer.\n\nFields:\n\nmmparams\n\n\n\n"
},

{
    "location": "net_components/layers.html#Public-Interface-1",
    "page": "Layers",
    "title": "Public Interface",
    "category": "section",
    "text": "Modules = [MIPVerify]\nOrder   = [:function, :type]\nPages   = [\n    \"net_components/layers/conv2d.jl\",\n    \"net_components/layers/convolution_layer.jl\",\n    \"net_components/layers/fully_connected_layer.jl\",\n    \"net_components/layers/masked_fully_connected_layer.jl\",\n    \"net_components/layers/matmul.jl\",\n    \"net_components/layers/pool.jl\",\n    \"net_components/layers/softmax.jl\"\n    ]\nPrivate = false"
},

{
    "location": "net_components/layers.html#MIPVerify.conv2d-Union{Tuple{Array{T,4},MIPVerify.Conv2DParameters{U,V}}, Tuple{T}, Tuple{U}, Tuple{V}} where V<:Union{JuMP.AbstractJuMPScalar, Real} where U<:Union{JuMP.AbstractJuMPScalar, Real} where T<:Union{JuMP.AbstractJuMPScalar, Real}",
    "page": "Layers",
    "title": "MIPVerify.conv2d",
    "category": "Method",
    "text": "conv2d(input, params)\n\n\nComputes the result of convolving input with the filter and bias stored in params.\n\nMirrors tf.nn.conv2d from the tensorflow package, with strides = [1, 1, 1, 1],  padding = \'SAME\'.\n\nThrows\n\nAssertionError if input and filter are not compatible.\n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.convolution_layer-Tuple{Array{#s110,4} where #s110<:Union{JuMP.AbstractJuMPScalar, Real},MIPVerify.ConvolutionLayerParameters}",
    "page": "Layers",
    "title": "MIPVerify.convolution_layer",
    "category": "Method",
    "text": "convolution_layer(x, params)\n\n\nComputes the result of convolving x with params.conv2dparams, pooling the resulting output with the pooling function and strides specified in params.maxpoolparams, and passing the output through a ReLU activation function.\n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.fully_connected_layer-Tuple{Array{#s109,1} where #s109<:JuMP.AbstractJuMPScalar,MIPVerify.FullyConnectedLayerParameters}",
    "page": "Layers",
    "title": "MIPVerify.fully_connected_layer",
    "category": "Method",
    "text": "Computes the result of multiplying x by params.mmparams, and  passing the output through a ReLU activation function.\n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.masked_fully_connected_layer-Tuple{Array{#s108,1} where #s108<:JuMP.AbstractJuMPScalar,MIPVerify.MaskedFullyConnectedLayerParameters}",
    "page": "Layers",
    "title": "MIPVerify.masked_fully_connected_layer",
    "category": "Method",
    "text": "Similar to fully_connected_layer(x, params), but with an additional mask in params that controls whether a ReLU is applied to each output. \n\nIf the value of the mask is <0 (i.e. input is assumed to be always non-positive), the  output is set at 0.\nIf the value of the mask is 0 (i.e. input can take both positive and negative values), the output is rectified.\nIf the value of the mask is >0 (i.e. input is assumed to be always non-negative), the  output is set as the value of the input, without any rectification.\n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.matmul-Tuple{Array{#s109,1} where #s109<:Union{JuMP.AbstractJuMPScalar, Real},MIPVerify.MatrixMultiplicationParameters}",
    "page": "Layers",
    "title": "MIPVerify.matmul",
    "category": "Method",
    "text": "matmul(x, params)\n\n\nComputes the result of pre-multiplying x by the transpose of params.matrix and adding params.bias.\n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.getoutputsize-Union{Tuple{AbstractArray{T,N},Tuple{Vararg{Int64,N}}}, Tuple{N}, Tuple{T}} where N where T",
    "page": "Layers",
    "title": "MIPVerify.getoutputsize",
    "category": "Method",
    "text": "getoutputsize(input_array, strides)\n\n\nFor pooling operations on an array, returns the expected size of the output array.\n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.getpoolview-Union{Tuple{AbstractArray{T,N},Tuple{Vararg{Int64,N}},Tuple{Vararg{Int64,N}}}, Tuple{N}, Tuple{T}} where N where T",
    "page": "Layers",
    "title": "MIPVerify.getpoolview",
    "category": "Method",
    "text": "getpoolview(input_array, strides, output_index)\n\n\nFor pooling operations on an array, returns a view of the parent array corresponding to the output_index in the output array.\n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.getsliceindex-Tuple{Int64,Int64,Int64}",
    "page": "Layers",
    "title": "MIPVerify.getsliceindex",
    "category": "Method",
    "text": "getsliceindex(input_array_size, stride, output_index)\n\n\nFor pooling operations on an array where a given element in the output array corresponds to equal-sized blocks in the input array, returns (for a given dimension) the index range in the input array corresponding to a particular index output_index in the output array.\n\nReturns an empty array if the output_index does not correspond to any input indices.\n\nArguments\n\nstride::Integer: the size of the operating blocks along the active    dimension.\n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.pool-Union{Tuple{AbstractArray{T,N},MIPVerify.PoolParameters{N}}, Tuple{N}, Tuple{T}} where N where T<:Union{JuMP.AbstractJuMPScalar, Real}",
    "page": "Layers",
    "title": "MIPVerify.pool",
    "category": "Method",
    "text": "pool(input, params)\n\n\nComputes the result of applying the pooling function params.pooling_function to  non-overlapping cells of input with sizes specified in params.strides.\n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.poolmap-Union{Tuple{Function,AbstractArray{T,N},Tuple{Vararg{Int64,N}}}, Tuple{N}, Tuple{T}} where N where T",
    "page": "Layers",
    "title": "MIPVerify.poolmap",
    "category": "Method",
    "text": "poolmap(f, input_array, strides)\n\n\nReturns output from applying f to subarrays of input_array, with the windows determined by the strides.\n\n\n\n"
},

{
    "location": "net_components/layers.html#Internal-1",
    "page": "Layers",
    "title": "Internal",
    "category": "section",
    "text": "Modules = [MIPVerify]\nOrder   = [:function, :type]\nPages   = [\n    \"net_components/layers/conv2d.jl\",\n    \"net_components/layers/convolution_layer.jl\",\n    \"net_components/layers/fully_connected_layer.jl\",\n    \"net_components/layers/masked_fully_connected_layer.jl\",\n    \"net_components/layers/matmul.jl\",\n    \"net_components/layers/pool.jl\",\n    \"net_components/layers/softmax.jl\"\n    ]\nPublic  = false"
},

{
    "location": "net_components/nets.html#",
    "page": "Networks",
    "title": "Networks",
    "category": "page",
    "text": ""
},

{
    "location": "net_components/nets.html#Networks-1",
    "page": "Networks",
    "title": "Networks",
    "category": "section",
    "text": "Each network corresponds to an array of layers associated with a unique string identifier. The string identifier of the network is used to store cached models, so it\'s important to ensure that you don\'t re-use names!"
},

{
    "location": "net_components/nets.html#MIPVerify.MaskedFullyConnectedNetParameters",
    "page": "Networks",
    "title": "MIPVerify.MaskedFullyConnectedNetParameters",
    "category": "Type",
    "text": "struct MaskedFullyConnectedNetParameters <: MIPVerify.NeuralNetParameters\n\nRepresents a neural net consisting of multiple masked fully-connected layers (as an array of MaskedFullyConnectedLayerParameters), followed by a softmax layer (as a  SoftmaxParameters).\n\nFields:\n\nmasked_fclayer_params\nsoftmax_params\nUUID\n\n\n\n"
},

{
    "location": "net_components/nets.html#MIPVerify.StandardNeuralNetParameters",
    "page": "Networks",
    "title": "MIPVerify.StandardNeuralNetParameters",
    "category": "Type",
    "text": "struct StandardNeuralNetParameters <: MIPVerify.NeuralNetParameters\n\nRepresents a neural net consisting of multiple convolution layers (as an array of ConvolutionLayerParameters), followed by multiple fully-connected layers (as an array of FullyConnectedLayerParameters), followed by a softmax layer (as a SoftmaxParameters).\n\nYou can leave the array convlayer_params empty if you do not have convolution layers, or conversely leave fclayers_empty if you do not have fully-connected layers. (Leaving _both_ empty doesn\'t make sense!)\n\nFields:\n\nconvlayer_params\nfclayer_params\nsoftmax_params\nUUID\n\n\n\n"
},

{
    "location": "net_components/nets.html#Public-Interface-1",
    "page": "Networks",
    "title": "Public Interface",
    "category": "section",
    "text": "Modules = [MIPVerify]\nOrder   = [:function, :type]\nPages   = [\n    \"net_components/nets/masked_fc_net.jl\",\n    \"net_components/nets/standard_neural_net.jl\",\n    ]\nPrivate = false"
},

{
    "location": "net_components/nets.html#Internal-1",
    "page": "Networks",
    "title": "Internal",
    "category": "section",
    "text": "Modules = [MIPVerify]\nOrder   = [:function, :type]\nPages   = [\n    \"net_components/nets/masked_fc_net.jl\",\n    \"net_components/nets/standard_neural_net.jl\",\n    ]\nPublic  = false"
},

{
    "location": "net_components/core_ops.html#",
    "page": "Core Operations",
    "title": "Core Operations",
    "category": "page",
    "text": ""
},

{
    "location": "net_components/core_ops.html#Core-Operations-1",
    "page": "Core Operations",
    "title": "Core Operations",
    "category": "section",
    "text": "Our ability to cast the input-output constraints of a neural net to an efficient set of linear and integer constraints boils down to the following basic operations, over which the layers provide a convenient layer of abstraction."
},

{
    "location": "net_components/core_ops.html#Index-1",
    "page": "Core Operations",
    "title": "Index",
    "category": "section",
    "text": "Pages   = [\"core_ops.md\"]\nOrder   = [:function, :type]"
},

{
    "location": "net_components/core_ops.html#MIPVerify.abs_ge-Tuple{JuMP.AbstractJuMPScalar}",
    "page": "Core Operations",
    "title": "MIPVerify.abs_ge",
    "category": "Method",
    "text": "abs_ge(x)\n\n\nExpresses a one-sided absolute-value constraint: output is constrained to be at least as large as |x|.\n\nOnly use when you are minimizing over the output in the objective.\n\n\n\n"
},

{
    "location": "net_components/core_ops.html#MIPVerify.masked_relu-Tuple{JuMP.AbstractJuMPScalar,Real}",
    "page": "Core Operations",
    "title": "MIPVerify.masked_relu",
    "category": "Method",
    "text": "masked_relu(x, m)\n\n\nExpresses a masked rectified-linearity constraint, with three possibilities depending on  the value of the mask. Output is constrained to be:\n\n1) max(x, 0) if m=0, \n2) 0 if m<0\n3) x if m>0\n\n\n\n"
},

{
    "location": "net_components/core_ops.html#MIPVerify.maximum-Union{Tuple{AbstractArray{T,N} where N}, Tuple{T}} where T<:JuMP.AbstractJuMPScalar",
    "page": "Core Operations",
    "title": "MIPVerify.maximum",
    "category": "Method",
    "text": "maximum(xs; tighten)\n\n\nExpresses a maximization constraint: output is constrained to be equal to max(xs).\n\n\n\n"
},

{
    "location": "net_components/core_ops.html#MIPVerify.relu-Tuple{JuMP.AbstractJuMPScalar}",
    "page": "Core Operations",
    "title": "MIPVerify.relu",
    "category": "Method",
    "text": "relu(x)\n\n\nExpresses a rectified-linearity constraint: output is constrained to be equal to  max(x, 0).\n\n\n\n"
},

{
    "location": "net_components/core_ops.html#MIPVerify.set_max_indexes-Tuple{Array{#s110,1} where #s110<:JuMP.AbstractJuMPScalar,Array{#s111,1} where #s111<:Integer}",
    "page": "Core Operations",
    "title": "MIPVerify.set_max_indexes",
    "category": "Method",
    "text": "set_max_indexes(x, target_indexes; tolerance)\n\n\nImposes constraints ensuring that one of the elements at the target_indexes is the  largest element of the array x. More specifically, we require x[j] - x[i] ≥ tolerance for some j ∈ target_indexes and for all i ∉ target_indexes.\n\n\n\n"
},

{
    "location": "net_components/core_ops.html#Internal-1",
    "page": "Core Operations",
    "title": "Internal",
    "category": "section",
    "text": "Modules = [MIPVerify]\nOrder   = [:function, :type]\nPages   = [\n    \"net_components/core_ops.jl\"\n    ]\nPublic  = false"
},

{
    "location": "utils/import_weights.html#",
    "page": "Importing Parameter Values",
    "title": "Importing Parameter Values",
    "category": "page",
    "text": ""
},

{
    "location": "utils/import_weights.html#Importing-Parameter-Values-1",
    "page": "Importing Parameter Values",
    "title": "Importing Parameter Values",
    "category": "section",
    "text": "You\'re likely to want to import parameter values from your trained neural networks from outside of Julia. get_example_network_params imports example networks provided as part of the package, while get_conv_params and get_matrix_params import individual layers."
},

{
    "location": "utils/import_weights.html#Index-1",
    "page": "Importing Parameter Values",
    "title": "Index",
    "category": "section",
    "text": "Pages   = [\"import_weights.md\"]\nOrder   = [:function, :type]"
},

{
    "location": "utils/import_weights.html#MIPVerify.get_conv_params-Tuple{Dict{String,V} where V,String,NTuple{4,Int64}}",
    "page": "Importing Parameter Values",
    "title": "MIPVerify.get_conv_params",
    "category": "Method",
    "text": "get_conv_params(param_dict, layer_name, expected_size; matrix_name, bias_name)\n\n\nHelper function to import the parameters for a convolution layer from param_dict as a     Conv2DParameters object.\n\nThe default format for parameter names is \'layer_name/weight\' and \'layer_name/bias\';      you can customize this by passing in the named arguments matrix_name and bias_name     respectively.\n\nArguments\n\nparam_dict::Dict{String}: Dictionary mapping parameter names to array of weights   / biases.\nlayer_name::String: Identifies parameter in dictionary.\nexpected_size::NTuple{4, Int}: Tuple of length 4 corresponding to the expected size   of the weights of the layer.\n\n\n\n"
},

{
    "location": "utils/import_weights.html#MIPVerify.get_example_network_params-Tuple{String}",
    "page": "Importing Parameter Values",
    "title": "MIPVerify.get_example_network_params",
    "category": "Method",
    "text": "get_example_network_params(name)\n\n\nMakes named example neural networks available as a NeuralNetParameters object.\n\nArguments\n\nname::String: Name of example neural network. Options:\n\'MNIST.n1\': MNIST classification. Two fully connected layers with 40 and 20   units, and softmax layer with 10 units. No adversarial training.\n\n\n\n"
},

{
    "location": "utils/import_weights.html#MIPVerify.get_matrix_params-Tuple{Dict{String,V} where V,String,Tuple{Int64,Int64}}",
    "page": "Importing Parameter Values",
    "title": "MIPVerify.get_matrix_params",
    "category": "Method",
    "text": "get_matrix_params(param_dict, layer_name, expected_size; matrix_name, bias_name)\n\n\nHelper function to import the parameters for a layer carrying out matrix multiplication      (e.g. fully connected layer / softmax layer) from param_dict as a     MatrixMultiplicationParameters object.\n\nThe default format for parameter names is \'layer_name/weight\' and \'layer_name/bias\';      you can customize this by passing in the named arguments matrix_name and bias_name     respectively.\n\nArguments\n\nparam_dict::Dict{String}: Dictionary mapping parameter names to array of weights   / biases.\nlayer_name::String: Identifies parameter in dictionary.\nexpected_size::NTuple{2, Int}: Tuple of length 2 corresponding to the expected size  of the weights of the layer.\n\n\n\n"
},

{
    "location": "utils/import_weights.html#Public-Interface-1",
    "page": "Importing Parameter Values",
    "title": "Public Interface",
    "category": "section",
    "text": "Modules = [MIPVerify]\nOrder   = [:function, :type]\nPages   = [\"utils/import_weights.jl\"]\nPrivate = false"
},

{
    "location": "utils/import_weights.html#Internal-1",
    "page": "Importing Parameter Values",
    "title": "Internal",
    "category": "section",
    "text": "Modules = [MIPVerify]\nOrder   = [:function, :type]\nPages   = [\"utils/import_weights.jl\"]\nPublic  = false"
},

{
    "location": "utils/import_datasets.html#",
    "page": "Importing External Datasets",
    "title": "Importing External Datasets",
    "category": "page",
    "text": ""
},

{
    "location": "utils/import_datasets.html#Importing-External-Datasets-1",
    "page": "Importing External Datasets",
    "title": "Importing External Datasets",
    "category": "section",
    "text": "For your convenience, the MNIST dataset is available as part of our package."
},

{
    "location": "utils/import_datasets.html#Index-1",
    "page": "Importing External Datasets",
    "title": "Index",
    "category": "section",
    "text": "Pages   = [\"import_datasets.md\"]\nOrder   = [:function, :type]"
},

{
    "location": "utils/import_datasets.html#MIPVerify.read_datasets-Tuple{String}",
    "page": "Importing External Datasets",
    "title": "MIPVerify.read_datasets",
    "category": "Method",
    "text": "read_datasets(name)\n\n\nMakes popular machine learning datasets available as a NamedTrainTestDataset.\n\nArguments\n\nname::String: name of machine learning dataset. Options:\nMNIST: The MNIST Database of handwritten digits\n\n\n\n"
},

{
    "location": "utils/import_datasets.html#Public-Interface-1",
    "page": "Importing External Datasets",
    "title": "Public Interface",
    "category": "section",
    "text": "Modules = [MIPVerify]\nOrder   = [:function, :type]\nPages   = [\"utils/import_datasets.jl\"]\nPrivate = false"
},

{
    "location": "utils/import_datasets.html#MIPVerify.ImageDataset",
    "page": "Importing External Datasets",
    "title": "MIPVerify.ImageDataset",
    "category": "Type",
    "text": "struct ImageDataset{T<:Real, U<:Int64} <: MIPVerify.Dataset\n\nDataset of images stored as a 4-dimensional array of size (num_samples, image_height,  image_width, num_channels), with accompanying labels (sorted in the same order) of size num_samples.\n\n\n\n"
},

{
    "location": "utils/import_datasets.html#MIPVerify.NamedTrainTestDataset",
    "page": "Importing External Datasets",
    "title": "MIPVerify.NamedTrainTestDataset",
    "category": "Type",
    "text": "struct NamedTrainTestDataset{T<:MIPVerify.Dataset} <: MIPVerify.Dataset\n\nNamed dataset containing a training set and a test set which are expected to contain the same kind of data.\n\n\n\n"
},

{
    "location": "utils/import_datasets.html#Internal-1",
    "page": "Importing External Datasets",
    "title": "Internal",
    "category": "section",
    "text": "Modules = [MIPVerify]\nOrder   = [:function, :type]\nPages   = [\"utils/import_datasets.jl\"]\nPublic  = false"
},

]}
