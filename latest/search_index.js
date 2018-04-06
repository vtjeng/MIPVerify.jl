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
    "text": "Discusses the various parameters you can select for find_adversarial_example. We explain how toBetter specify targeted labels for the perturbed image (including multiple targeted labels)\nHave more precise control over the activations in the output layer\nRestrict the family of perturbations (for example to the blurring perturbations discussed in our paper)\nSelect whether you want to minimize the L_1, L_2 or L_infty norm of the perturbation.\nDetermine whether you are rebuilding the model expressing the constraints of the neural network from scratch, or loading the model from cache.\nModify the amount of time dedicated to building the model (by selecting the tightening_algorithm, and/or passing in a custom tightening_solver).For Gurobi, we show how to specify solver settings to:Mute output\nTerminate early if:\nA time limit is reached\nLower bounds on robustness are proved (that is, we prove that no adversarial example can exist closer than some threshold)\nAn adversarial example is found that is closer to the input than expected\nThe gap between the upper and lower objective bounds falls below a selected threshold"
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
    "text": "Explains how to get more granular log settings and to write log output to file."
},

{
    "location": "finding_adversarial_examples/basic_usage.html#",
    "page": "Basic Usage",
    "title": "Basic Usage",
    "category": "page",
    "text": ""
},

{
    "location": "finding_adversarial_examples/basic_usage.html#Basic-Usage-1",
    "page": "Basic Usage",
    "title": "Basic Usage",
    "category": "section",
    "text": "find_adversarial_example is the core function that you will be calling to  find adversarial examples. To avoid spending time verifying the wrong network, we suggest that you check that the network gets reasonable performance on the test set using frac_correct."
},

{
    "location": "finding_adversarial_examples/basic_usage.html#Index-1",
    "page": "Basic Usage",
    "title": "Index",
    "category": "section",
    "text": "Pages   = [\"basic_usage.md\"]\nOrder   = [:function, :type]"
},

{
    "location": "finding_adversarial_examples/basic_usage.html#MIPVerify.find_adversarial_example-Tuple{MIPVerify.NeuralNet,Array{#s26,N} where N where #s26<:Real,Union{Array{#s25,1} where #s25<:Integer, Integer},MathProgBase.SolverInterface.AbstractMathProgSolver}",
    "page": "Basic Usage",
    "title": "MIPVerify.find_adversarial_example",
    "category": "method",
    "text": "find_adversarial_example(nn, input, target_selection, main_solver; pp, norm_order, tolerance, rebuild, invert_target_selection, tightening_algorithm, tightening_solver, cache_model)\n\n\nFinds the perturbed image closest to input such that the network described by nn classifies the perturbed image in one of the categories identified by the  indexes in target_selection.\n\nmain_solver specifies the solver used.\n\nFormal Definition: If there are a total of n categories, the output vector y has  length n. We guarantee that y[j] - y[i] ≥ tolerance for some j ∈ target_selection  and for all i ∉ target_selection.\n\nNamed Arguments:\n\npp::PerturbationFamily: Defaults to UnrestrictedPerturbationFamily(). Determines   the family of perturbations over which we are searching for adversarial examples.\nnorm_order::Real: Defaults to 1. Determines the distance norm used to determine the    distance from the perturbed image to the original. Supported options are 1, Inf    and 2 (if the main_solver used can solve MIQPs.)\ntolerance::Real: Defaults to 0.0. See formal definition above.\nrebuild::Bool: Defaults to false. If true, rebuilds model by determining upper and lower   bounds on input to each non-linear unit even if a cached model exists.\ninvert_target_selection::Bool: Defaults to false. If true, sets target_selection to    be its complement.\ntightening_algorithm::MIPVerify.TighteningAlgorithm: Defaults to lp. Determines how we determine the upper and lower   bounds on input to each nonlinear unit. Allowed options are interval_arithmetic, lp, mip.  (1) interval_arithmetic looks at the bounds on the output to the previous layer.  (2) lp solves an lp corresponding to the mip formulation, but with any integer constraints relaxed.  (3) mip solves the full mip formulation.\ntightening_solver: Defaults to the same type of solver as   the main_solver, with a time limit of 20s per solver and output suppressed. Used only   if the tightening_algorithm is lp or mip.\ncache_model: Defaults to true. If true, saves model generated (but does not remove)   existing cached model if false.\n\n\n\n"
},

{
    "location": "finding_adversarial_examples/basic_usage.html#MIPVerify.frac_correct-Tuple{MIPVerify.NeuralNet,MIPVerify.LabelledDataset,Int64}",
    "page": "Basic Usage",
    "title": "MIPVerify.frac_correct",
    "category": "method",
    "text": "frac_correct(nn, dataset, num_samples)\n\n\nReturns the fraction of items the neural network correctly classifies of the first num_samples of the provided dataset. If there are fewer than num_samples items, we use all of the available samples.\n\nNamed Arguments:\n\nnn::NeuralNet: The parameters of the neural network.\ndataset::LabelledDataset:\nnum_samples::Int: Number of samples to use.\n\n\n\n"
},

{
    "location": "finding_adversarial_examples/basic_usage.html#Public-Interface-1",
    "page": "Basic Usage",
    "title": "Public Interface",
    "category": "section",
    "text": "Modules = [MIPVerify]\nOrder   = [:function, :type]\nPages   = [\"MIPVerify.jl\"]\nPrivate = false"
},

{
    "location": "finding_adversarial_examples/batch_processing.html#",
    "page": "Batch Processing",
    "title": "Batch Processing",
    "category": "page",
    "text": ""
},

{
    "location": "finding_adversarial_examples/batch_processing.html#Batch-Processing-1",
    "page": "Batch Processing",
    "title": "Batch Processing",
    "category": "section",
    "text": "When running on multiple samples from a single dataset, batch_find_certificate  allows you to redo solves intelligently - redoing 1) no solves, 2) all solves, 3) only  solves where the sample status is indeterminate, or 4) only solves where the best counter-example is non-optimal."
},

{
    "location": "finding_adversarial_examples/batch_processing.html#Index-1",
    "page": "Batch Processing",
    "title": "Index",
    "category": "section",
    "text": "Pages   = [\"batch_processing.md\"]\nOrder   = [:function, :type]"
},

{
    "location": "finding_adversarial_examples/batch_processing.html#MIPVerify.batch_find_certificate-Tuple{MIPVerify.NeuralNet,MIPVerify.LabelledDataset,AbstractArray{#s29,N} where N where #s29<:Integer,MathProgBase.SolverInterface.AbstractMathProgSolver}",
    "page": "Batch Processing",
    "title": "MIPVerify.batch_find_certificate",
    "category": "method",
    "text": "batch_find_certificate(nn, dataset, target_sample_numbers, main_solver; save_path, pp, norm_order, tolerance, rebuild, tightening_algorithm, tightening_solver, solve_rerun_option, cache_model)\n\n\nRuns find_adversarial_example for the specified neural network nn and dataset for the target_sample_numbers, skipping target_sample_numbers based on the selected solve_rerun_option.\n\nIt creates a named directory in save_path, with the name summarizing \n\nthe name of the network in nn, \nthe perturbation family pp, \nthe norm_order\nthe tolerance.\n\nWithin this directory, a summary of all the results is stored in summary.csv, and  results from individual runs are stored in the subfolder run_results.\n\nThis functioned is designed so that it can be interrupted and restarted cleanly; it relies on the summary.csv file to determine what the results of previous runs are (so modifying this file manually can lead to unexpected behavior.)\n\nmain_solver specifies the solver used.\n\nNamed Arguments:\n\npp, norm_order, tolerance, rebuild, tightening_algorithm, tightening_solver are passed directly to find_adversarial_example; see that documentation for more details.\n\npp::PerturbationFamily: Defaults to UnrestrictedPerturbationFamily(). \nnorm_order::Real: Defaults to 1.\ntolerance::Real: Defaults to 0.0.\nrebuild::Bool: Defaults to false.\ntightening_algorithm::MIPVerify.TighteningAlgorithm: Defaults to lp.\ntightening_solver: \nsolve_rerun_option::MIPVerify.SolveRerunOption: Options are  never, always, resolve_ambiguous_cases, and refine_insecure_cases.  See run_on_sample for more details.\n\n\n\n"
},

{
    "location": "finding_adversarial_examples/batch_processing.html#Public-Interface-1",
    "page": "Batch Processing",
    "title": "Public Interface",
    "category": "section",
    "text": "Modules = [MIPVerify]\nOrder   = [:function, :type]\nPages   = [\"batch_processing_helpers.jl\"]\nPrivate = false"
},

{
    "location": "finding_adversarial_examples/batch_processing.html#MIPVerify.run_on_sample-Tuple{Int64,DataFrames.DataFrame,MIPVerify.SolveRerunOption}",
    "page": "Batch Processing",
    "title": "MIPVerify.run_on_sample",
    "category": "method",
    "text": "run_on_sample(sample_number, summary_dt, solve_rerun_option)\n\n\nDetermines whether to run a solve on a sample depending on the solve_rerun_option by looking up information on the most recent completed solve. recorded in summary_dt\n\nsummary_dt is expected to be a DataFrame with columns :SampleNumber, :SolveStatus, and :ObjectiveValue. \n\nBehavior for different choices of solve_rerun_option:\n\nnever: true if and only if there is no previous completed solve.\nalways: true always.\nresolve_ambiguous_cases: true if there is no previous completed solve, or if the    most recent completed solve a) did not find a counter-example BUT b) the optimization   was not demosntrated to be infeasible.\nrefine_insecure_cases: true if there is no previous completed solve, or if the most   recent complete solve a) did find a counter-example BUT b) we did not reach a    provably optimal solution.\n\n\n\n"
},

{
    "location": "finding_adversarial_examples/batch_processing.html#Internal-1",
    "page": "Batch Processing",
    "title": "Internal",
    "category": "section",
    "text": "Modules = [MIPVerify]\nOrder   = [:function, :type]\nPages   = [\"batch_processing_helpers.jl\"]\nPublic  = false"
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
    "text": "A neural net consists of multiple layers, each of which (potentially) operates on input differently. We represent these objects with NeuralNet and Layer."
},

{
    "location": "net_components/overview.html#Index-1",
    "page": "Overview",
    "title": "Index",
    "category": "section",
    "text": "Pages   = [\"overview.md\"]\nOrder   = [:function, :type]"
},

{
    "location": "net_components/overview.html#MIPVerify.Layer",
    "page": "Overview",
    "title": "MIPVerify.Layer",
    "category": "type",
    "text": "abstract Layer\n\nSupertype for all types storing the parameters of each layer. Inherit from this to specify your own custom type of layer. Each implementation is expected to:\n\nImplement a callable specifying the output when any input of type JuMPReal is provided.\n\n\n\n"
},

{
    "location": "net_components/overview.html#MIPVerify.NeuralNet",
    "page": "Overview",
    "title": "MIPVerify.NeuralNet",
    "category": "type",
    "text": "abstract NeuralNet\n\nSupertype for all types storing the parameters of a neural net. Inherit from this to specify your own custom architecture. Each implementation is expected to:\n\nImplement a callable specifying the output when any input of type JuMPReal is provided\nHave a UUID field for the name of the neural network.\n\n\n\n"
},

{
    "location": "net_components/overview.html#MIPVerify.chain-Tuple{Array{#s120,N} where N where #s120<:Union{JuMP.GenericAffExpr{Float64,JuMP.Variable}, JuMP.Variable, Real},Array{#s119,1} where #s119<:MIPVerify.Layer}",
    "page": "Overview",
    "title": "MIPVerify.chain",
    "category": "method",
    "text": "An array of Layers is interpreted as that array of layer being applied to the input sequentially, starting from the leftmost layer. (In functional programming terms, this can be thought of as a sort of fold).\n\n\n\n"
},

{
    "location": "net_components/overview.html#PublicInterface-1",
    "page": "Overview",
    "title": "PublicInterface",
    "category": "section",
    "text": "Modules = [MIPVerify]\nOrder   = [:function, :type]\nPages   = [\"net_components.jl\"]"
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
    "location": "net_components/layers.html#MIPVerify.Conv2d",
    "page": "Layers",
    "title": "MIPVerify.Conv2d",
    "category": "type",
    "text": "struct Conv2d{T<:Union{JuMP.GenericAffExpr{Float64,JuMP.Variable}, JuMP.Variable, Real}, U<:Union{JuMP.GenericAffExpr{Float64,JuMP.Variable}, JuMP.Variable, Real}, V<:Int64} <: MIPVerify.Layer\n\nRepresents 2-D convolution operation.\n\np(x) is shorthand for conv2d(x, p) when p is an instance of Conv2d.\n\nFields:\n\nfilter\nbias\nstride\n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.Conv2d-Union{Tuple{Array{T,4}}, Tuple{T}} where T<:Union{JuMP.GenericAffExpr{Float64,JuMP.Variable}, JuMP.Variable, Real}",
    "page": "Layers",
    "title": "MIPVerify.Conv2d",
    "category": "method",
    "text": "Conv2d(filter)\n\n\nConvenience function to create a Conv2d struct with the specified filter and zero bias.\n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.Flatten",
    "page": "Layers",
    "title": "MIPVerify.Flatten",
    "category": "type",
    "text": "struct Flatten{T<:Int64} <: MIPVerify.Layer\n\nRepresents a flattening operation.\n\np(x) is shorthand for flatten(x, p.perm) when p is an instance of Flatten.\n\nFields:\n\nn_dim\nperm\n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.Linear",
    "page": "Layers",
    "title": "MIPVerify.Linear",
    "category": "type",
    "text": "struct Linear{T<:Real, U<:Real} <: MIPVerify.Layer\n\nRepresents matrix multiplication.\n\np(x) is shorthand for matmul(x, p) when p is an instance of Linear.\n\nFields:\n\nmatrix\nbias\n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.MaskedReLU",
    "page": "Layers",
    "title": "MIPVerify.MaskedReLU",
    "category": "type",
    "text": "struct MaskedReLU{T<:Real} <: MIPVerify.Layer\n\nRepresents a masked ReLU activation, with mask controlling how the ReLU is applied to each output.\n\np(x) is shorthand for masked_relu(x, p.mask) when p is an instance of MaskedReLU.\n\nFields:\n\nmask\n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.MaxPool-Union{Tuple{N}, Tuple{Tuple{Vararg{Int64,N}}}} where N",
    "page": "Layers",
    "title": "MIPVerify.MaxPool",
    "category": "method",
    "text": "MaxPool(strides)\n\n\nConvenience function to create a Pool struct for max-pooling.\n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.Pool",
    "page": "Layers",
    "title": "MIPVerify.Pool",
    "category": "type",
    "text": "struct Pool{N} <: MIPVerify.Layer\n\nRepresents a pooling operation.\n\np(x) is shorthand for pool(x, p) when p is an instance of Pool.\n\nFields:\n\nstrides\npooling_function\n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.ReLU",
    "page": "Layers",
    "title": "MIPVerify.ReLU",
    "category": "type",
    "text": "struct ReLU <: MIPVerify.Layer\n\nRepresents a ReLU operation.\n\np(x) is shorthand for relu(x) when p is an instance of ReLU.\n\n\n\n"
},

{
    "location": "net_components/layers.html#Public-Interface-1",
    "page": "Layers",
    "title": "Public Interface",
    "category": "section",
    "text": "Modules = [MIPVerify]\nOrder   = [:function, :type]\nPages   = [\n    \"net_components/layers/conv2d.jl\",\n    \"net_components/layers/flatten.jl\",\n    \"net_components/layers/linear.jl\",\n    \"net_components/layers/masked_relu.jl\",\n    \"net_components/layers/pool.jl\",\n    \"net_components/layers/relu.jl\"\n    ]\nPrivate = false"
},

{
    "location": "net_components/layers.html#MIPVerify.conv2d-Union{Tuple{Array{T,4},MIPVerify.Conv2d{U,V,V} where V<:Int64}, Tuple{T}, Tuple{U}, Tuple{V}} where V<:Union{JuMP.GenericAffExpr{Float64,JuMP.Variable}, JuMP.Variable, Real} where U<:Union{JuMP.GenericAffExpr{Float64,JuMP.Variable}, JuMP.Variable, Real} where T<:Union{JuMP.GenericAffExpr{Float64,JuMP.Variable}, JuMP.Variable, Real}",
    "page": "Layers",
    "title": "MIPVerify.conv2d",
    "category": "method",
    "text": "conv2d(input, params)\n\n\nComputes the result of convolving input with the filter and bias stored in params.\n\nMirrors tf.nn.conv2d from the tensorflow package, with strides = [1, 1, 1, 1],  padding = \'SAME\'.\n\nThrows\n\nAssertionError if input and filter are not compatible.\n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.flatten-Union{Tuple{Array{T,N},AbstractArray{U,N} where N}, Tuple{N}, Tuple{T}, Tuple{U}} where U<:Int64 where N where T",
    "page": "Layers",
    "title": "MIPVerify.flatten",
    "category": "method",
    "text": "Permute dimensions of array in specified order, then flattens the array.\n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.matmul-Tuple{Array{#s119,1} where #s119<:Union{JuMP.GenericAffExpr{Float64,JuMP.Variable}, JuMP.Variable, Real},MIPVerify.Linear}",
    "page": "Layers",
    "title": "MIPVerify.matmul",
    "category": "method",
    "text": "matmul(x, params)\n\n\nComputes the result of pre-multiplying x by the transpose of params.matrix and adding params.bias.\n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.getoutputsize-Union{Tuple{AbstractArray{T,N},Tuple{Vararg{Int64,N}}}, Tuple{N}, Tuple{T}} where N where T",
    "page": "Layers",
    "title": "MIPVerify.getoutputsize",
    "category": "method",
    "text": "getoutputsize(input_array, strides)\n\n\nFor pooling operations on an array, returns the expected size of the output array.\n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.getpoolview-Union{Tuple{AbstractArray{T,N},Tuple{Vararg{Int64,N}},Tuple{Vararg{Int64,N}}}, Tuple{N}, Tuple{T}} where N where T",
    "page": "Layers",
    "title": "MIPVerify.getpoolview",
    "category": "method",
    "text": "getpoolview(input_array, strides, output_index)\n\n\nFor pooling operations on an array, returns a view of the parent array corresponding to the output_index in the output array.\n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.getsliceindex-Tuple{Int64,Int64,Int64}",
    "page": "Layers",
    "title": "MIPVerify.getsliceindex",
    "category": "method",
    "text": "getsliceindex(input_array_size, stride, output_index)\n\n\nFor pooling operations on an array where a given element in the output array corresponds to equal-sized blocks in the input array, returns (for a given dimension) the index range in the input array corresponding to a particular index output_index in the output array.\n\nReturns an empty array if the output_index does not correspond to any input indices.\n\nArguments\n\nstride::Integer: the size of the operating blocks along the active    dimension.\n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.pool-Union{Tuple{AbstractArray{T,N},MIPVerify.Pool{N}}, Tuple{N}, Tuple{T}} where N where T<:Union{JuMP.GenericAffExpr{Float64,JuMP.Variable}, JuMP.Variable, Real}",
    "page": "Layers",
    "title": "MIPVerify.pool",
    "category": "method",
    "text": "pool(input, params)\n\n\nComputes the result of applying the pooling function params.pooling_function to  non-overlapping cells of input with sizes specified in params.strides.\n\n\n\n"
},

{
    "location": "net_components/layers.html#MIPVerify.poolmap-Union{Tuple{Function,AbstractArray{T,N},Tuple{Vararg{Int64,N}}}, Tuple{N}, Tuple{T}} where N where T",
    "page": "Layers",
    "title": "MIPVerify.poolmap",
    "category": "method",
    "text": "poolmap(f, input_array, strides)\n\n\nReturns output from applying f to subarrays of input_array, with the windows determined by the strides.\n\n\n\n"
},

{
    "location": "net_components/layers.html#Internal-1",
    "page": "Layers",
    "title": "Internal",
    "category": "section",
    "text": "Modules = [MIPVerify]\nOrder   = [:function, :type]\nPages   = [\n    \"net_components/layers/conv2d.jl\",\n    \"net_components/layers/flatten.jl\",\n    \"net_components/layers/linear.jl\",\n    \"net_components/layers/masked_relu.jl\",\n    \"net_components/layers/pool.jl\",\n    \"net_components/layers/relu.jl\"\n    ]\nPublic  = false"
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
    "location": "net_components/nets.html#MIPVerify.Sequential",
    "page": "Networks",
    "title": "MIPVerify.Sequential",
    "category": "type",
    "text": "struct Sequential <: MIPVerify.NeuralNet\n\nRepresents a sequential (feed-forward) neural net, with layers ordered from input to output.\n\nFields:\n\nlayers\nUUID\n\n\n\n"
},

{
    "location": "net_components/nets.html#Public-Interface-1",
    "page": "Networks",
    "title": "Public Interface",
    "category": "section",
    "text": "Modules = [MIPVerify]\nOrder   = [:function, :type]\nPages   = [\n    \"net_components/nets/sequential.jl\",\n    ]\nPrivate = false"
},

{
    "location": "net_components/nets.html#Internal-1",
    "page": "Networks",
    "title": "Internal",
    "category": "section",
    "text": "Modules = [MIPVerify]\nOrder   = [:function, :type]\nPages   = [\n    \"net_components/nets/sequential.jl\",\n    ]\nPublic  = false"
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
    "location": "net_components/core_ops.html#MIPVerify.abs_ge-Tuple{Union{JuMP.GenericAffExpr{Float64,JuMP.Variable}, JuMP.Variable}}",
    "page": "Core Operations",
    "title": "MIPVerify.abs_ge",
    "category": "method",
    "text": "abs_ge(x)\n\n\nExpresses a one-sided absolute-value constraint: output is constrained to be at least as large as |x|.\n\nOnly use when you are minimizing over the output in the objective.\n\n\n\n"
},

{
    "location": "net_components/core_ops.html#MIPVerify.masked_relu-Tuple{AbstractArray{#s31,N} where N where #s31<:Union{JuMP.GenericAffExpr{Float64,JuMP.Variable}, JuMP.Variable},AbstractArray{#s30,N} where N where #s30<:Real}",
    "page": "Core Operations",
    "title": "MIPVerify.masked_relu",
    "category": "method",
    "text": "masked_relu(x, m)\n\n\nExpresses a masked rectified-linearity constraint, with three possibilities depending on  the value of the mask. Output is constrained to be:\n\n1) max(x, 0) if m=0, \n2) 0 if m<0\n3) x if m>0\n\n\n\n"
},

{
    "location": "net_components/core_ops.html#MIPVerify.maximum-Union{Tuple{AbstractArray{T,N} where N}, Tuple{T}} where T<:Union{JuMP.GenericAffExpr{Float64,JuMP.Variable}, JuMP.Variable}",
    "page": "Core Operations",
    "title": "MIPVerify.maximum",
    "category": "method",
    "text": "maximum(xs; tightening_algorithm)\n\n\nExpresses a maximization constraint: output is constrained to be equal to max(xs).\n\n\n\n"
},

{
    "location": "net_components/core_ops.html#MIPVerify.maximum_ge-Union{Tuple{AbstractArray{T,N} where N}, Tuple{T}} where T<:Union{JuMP.GenericAffExpr{Float64,JuMP.Variable}, JuMP.Variable}",
    "page": "Core Operations",
    "title": "MIPVerify.maximum_ge",
    "category": "method",
    "text": "maximum_ge(xs)\n\n\nExpresses a one-sided maximization constraint: output is constrained to be at least  max(xs).\n\nOnly use when you are minimizing over the output in the objective.\n\n\n\n"
},

{
    "location": "net_components/core_ops.html#MIPVerify.relu-Union{Tuple{AbstractArray{T,N} where N}, Tuple{T}} where T<:Union{JuMP.GenericAffExpr{Float64,JuMP.Variable}, JuMP.Variable}",
    "page": "Core Operations",
    "title": "MIPVerify.relu",
    "category": "method",
    "text": "relu(x)\nrelu(x)\n\n\nExpresses a rectified-linearity constraint: output is constrained to be equal to  max(x, 0).\n\n\n\n"
},

{
    "location": "net_components/core_ops.html#MIPVerify.set_max_indexes-Tuple{Array{#s120,1} where #s120<:Union{JuMP.GenericAffExpr{Float64,JuMP.Variable}, JuMP.Variable},Array{#s121,1} where #s121<:Integer}",
    "page": "Core Operations",
    "title": "MIPVerify.set_max_indexes",
    "category": "method",
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
    "category": "method",
    "text": "get_conv_params(param_dict, layer_name, expected_size; matrix_name, bias_name, expected_stride)\n\n\nHelper function to import the parameters for a convolution layer from param_dict as a     Conv2d object.\n\nThe default format for parameter names is \'layer_name/weight\' and \'layer_name/bias\';      you can customize this by passing in the named arguments matrix_name and bias_name     respectively.\n\nArguments\n\nparam_dict::Dict{String}: Dictionary mapping parameter names to array of weights   / biases.\nlayer_name::String: Identifies parameter in dictionary.\nexpected_size::NTuple{4, Int}: Tuple of length 4 corresponding to the expected size   of the weights of the layer.\n\n\n\n"
},

{
    "location": "utils/import_weights.html#MIPVerify.get_example_network_params-Tuple{String}",
    "page": "Importing Parameter Values",
    "title": "MIPVerify.get_example_network_params",
    "category": "method",
    "text": "get_example_network_params(name)\n\n\nMakes named example neural networks available as a NeuralNet object.\n\nArguments\n\nname::String: Name of example neural network. Options:\n\'MNIST.n1\': MNIST classification. Two fully connected layers with 40 and 20   units, and softmax layer with 10 units. No adversarial training.\n\n\n\n"
},

{
    "location": "utils/import_weights.html#MIPVerify.get_matrix_params-Tuple{Dict{String,V} where V,String,Tuple{Int64,Int64}}",
    "page": "Importing Parameter Values",
    "title": "MIPVerify.get_matrix_params",
    "category": "method",
    "text": "get_matrix_params(param_dict, layer_name, expected_size; matrix_name, bias_name)\n\n\nHelper function to import the parameters for a layer carrying out matrix multiplication      (e.g. fully connected layer / softmax layer) from param_dict as a     Linear object.\n\nThe default format for parameter names is \'layer_name/weight\' and \'layer_name/bias\';      you can customize this by passing in the named arguments matrix_name and bias_name     respectively.\n\nArguments\n\nparam_dict::Dict{String}: Dictionary mapping parameter names to array of weights   / biases.\nlayer_name::String: Identifies parameter in dictionary.\nexpected_size::NTuple{2, Int}: Tuple of length 2 corresponding to the expected size  of the weights of the layer.\n\n\n\n"
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
    "category": "method",
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
    "location": "utils/import_datasets.html#MIPVerify.LabelledImageDataset",
    "page": "Importing External Datasets",
    "title": "MIPVerify.LabelledImageDataset",
    "category": "type",
    "text": "struct LabelledImageDataset{T<:Real, U<:Int64} <: MIPVerify.LabelledDataset\n\nDataset of images stored as a 4-dimensional array of size (num_samples, image_height,  image_width, num_channels), with accompanying labels (sorted in the same order) of size num_samples.\n\n\n\n"
},

{
    "location": "utils/import_datasets.html#MIPVerify.NamedTrainTestDataset",
    "page": "Importing External Datasets",
    "title": "MIPVerify.NamedTrainTestDataset",
    "category": "type",
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
