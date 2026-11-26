repository = get(ENV, "WK17_REPO", pwd())
include(joinpath(repository, "benchmarks", "backward_margin_screen.jl"))
using MIPVerify: Conv2d, Linear, ReLU, Flatten
const Q=Rational{BigInt}
exact(p::Conv2d)=Conv2d(Q.(p.filter),Q.(p.bias),p.stride,p.padding)
exact(p::Linear)=Linear(Q.(p.matrix),Q.(p.bias))
exact(p::Union{ReLU,Flatten})=p
function replay()
    nn=MIPVerify.get_example_network_params("MNIST.WK17a_linf0.1_authors")
    net=exact.(nn.layers)
    test=MIPVerify.read_datasets("MNIST").test
    stages=compile_stages(nn,size(MIPVerify.get_image(test,1)))
    println("image,hidden_bounds_checked,strict_margin,winning_digit")
    input_path = isempty(ARGS) ? joinpath(@__DIR__, "confirmed-witnesses.csv") : only(ARGS)
    for r in CSV.File(input_path)
        (ismissing(r.perturbed_input_value) || isempty(r.perturbed_input_value)) && continue
        x=MIPVerify.get_image(test,r.sample_index)
        y=MIPVerify.get_label(test,r.sample_index)+1
        z=reshape(Q.(parse.(Float64,split(r.perturbed_input_value,';'))),size(x))
        h,lo,hi=hidden_bounds(stages,x)
        lower=max.(0,round.(Int,255 .* vec(x)).//255 .- 1//10)
        upper=min.(1,round.(Int,255 .* vec(x)).//255 .+ 1//10)
        @assert all(Q.(lo).<=lower) && all(Q.(hi).>=upper)
        @assert all(lower .<= vec(z) .<= upper) && all(abs.(Q.(x).-z).<=1//10)
        v=z; layer=0; checked=0
        for p in net
            if p isa ReLU
                layer+=1
                l,u=h[layer]
                # Use the network's native tensor order and original rational layer implementation.
                @assert all(Q.(l) .<= vec(v) .<= Q.(u)) (r.sample_index,layer)
                checked+=length(v)
            end
            v=p(v)
        end
        gap=maximum(v[setdiff(eachindex(v),y)])-v[y]
        @assert gap>0
        println(r.sample_index,",",checked,",",Float64(gap),",",argmax(v)-1)
        flush(stdout)
    end
end
replay()
