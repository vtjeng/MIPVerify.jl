repository = get(ENV, "WK17_REPO", pwd())
include(joinpath(repository, "benchmarks", "backward_margin_screen.jl"))
zero_bias(p::MIPVerify.Conv2d)=MIPVerify.Conv2d(p.filter,zero(p.bias),p.stride,p.padding)
zero_bias(p::MIPVerify.Linear)=MIPVerify.Linear(p.matrix,zero(p.bias))
function audit()
    nn=MIPVerify.get_example_network_params("MNIST.WK17a_linf0.1_authors")
    shape=(1,28,28,1);stages=compile_stages(nn,shape);k=0
    flatten=identity
    for p in nn.layers
        if p isa MIPVerify.Flatten
            flatten=p
        elseif !(p isa MIPVerify.ReLU)
            k+=1;s=stages[k];raw=zeros(shape);unbiased=zero_bias(p)
            bias=p(flatten(raw))
            @assert vec(bias)==s.bias
            W=Matrix(s.weights)
            # One nonzero input and zero biases make each native evaluation
            # an exact copy of one weight or zero, including padding and flatten.
            for i in eachindex(raw)
                raw[i]=1
                @assert vec(unbiased(flatten(raw)))==view(W,i,:) (k,i)
                raw[i]=0
            end
            println("stage=",k," bases=",length(raw)," coefficients=",length(W)," biases=",length(bias));flush(stdout)
            shape=size(bias);flatten=identity
        end
    end
end
audit()
