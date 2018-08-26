# Batch Processing
[`batch_find_untargeted_attack`](@ref) enables users to run [`find_adversarial_example`](@ref) 
for multiple samples from a single dataset, writing 1) a single summary `.csv` file 
for the dataset, with a row of summary results per sample, and 2) a file per sample containing the
output dictionary from [`find_adversarial_example`](@ref).

[`batch_find_untargeted_attack`](@ref) allows verification of a dataset to be resumed if the
process is interrupted by intelligently determining whether to rerun [`find_adversarial_example`](@ref)
on a sample based on the `solve_rerun_option` specified.

## Index
```@index
Pages   = ["batch_processing.md"]
Order   = [:function, :type]
```

## Public Interface
```@autodocs
Modules = [MIPVerify]
Order   = [:function, :type]
Pages   = ["batch_processing_helpers.jl"]
Private = false
```

## Internal
```@autodocs
Modules = [MIPVerify]
Order   = [:function, :type]
Pages   = ["batch_processing_helpers.jl"]
Public  = false
```