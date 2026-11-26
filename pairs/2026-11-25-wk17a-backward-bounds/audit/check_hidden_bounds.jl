# Independent real-arithmetic checker of the bounds the floating verifier actually supplied.
repository = get(ENV, "WK17_REPO", pwd())
include(joinpath(repository, "benchmarks", "backward_margin_screen.jl"))
const Exact = Rational{BigInt}
struct ExactLayer
    W::SparseMatrixCSC{Exact,Int}
    b::Vector{Exact}
end
function boxsum(c,b,lo,hi)
    for i in eachindex(c)
        iszero(c[i]) && continue
        b+=c[i]*(c[i]>0 ? hi[i] : lo[i])
    end
    return b
end
function affineinterval(layer,lo,hi,j)
    l=u=layer.b[j]
    for k in nzrange(layer.W,j)
        w=nonzeros(layer.W)[k]; i=rowvals(layer.W)[k]
        l+=w*(w>0 ? lo[i] : hi[i])
        u+=w*(w>0 ? hi[i] : lo[i])
    end
    return l,u
end
function upper(layers,k,c,hidden,box)
    b=zero(Exact)
    for s in k:-1:1
        W=layers[s].W
        prev=zeros(Exact,size(W,1))
        for j in eachindex(c)
            a=c[j]; iszero(a) && continue
            b+=a*layers[s].b[j]
            for z in nzrange(W,j)
                prev[rowvals(W)[z]]+=a*nonzeros(W)[z]
            end
        end
        if s>1
            lo,hi,frac=hidden[s-1]
            for i in eachindex(prev)
                a,l,u=prev[i],lo[i],hi[i]
                if iszero(a) || l>=0
                    continue
                elseif u<=0
                    prev[i]=0
                elseif a>0
                    # Any dyadic slope in [0,a] works. Independently enclose both
                    # endpoints in exact arithmetic (the chord dominates a convex ReLU).
                    slope=a*frac[i]
                    b+=max(-slope*l,(a-slope)*u)
                    prev[i]=slope
                else
                    prev[i]=u < -l ? 0 : a
                end
            end
        end
        c=prev
    end
    return boxsum(c,b,box...)
end
function audit(i)
    nn=MIPVerify.get_example_network_params("MNIST.WK17a_linf0.1_authors")
    test=MIPVerify.read_datasets("MNIST").test
    x=MIPVerify.get_image(test,i); y=MIPVerify.get_label(test,i)+1
    stages=compile_stages(nn,size(x))
    floats,lo,hi=hidden_bounds(stages,x)
    box=(Exact.(lo),Exact.(hi))
    # An enclosure of both the stored floating centers and the original normalized bytes.
    for center in [Exact.(vec(x)), round.(Int,vec(x).*255).//255]
        @assert all(box[1] .<= max.(0,center .- 1//10)) && all(box[2] .>= min.(1,center .+ 1//10))
    end
    @assert all(lo .<= max.(0,vec(x) .- 0.1)) && all(hi .>= min.(1,vec(x) .+ 0.1))
    layers=[ExactLayer(Exact.(s.weights),Exact.(s.bias)) for s in stages]
    hidden=Tuple{Vector{Exact},Vector{Exact},Vector{Exact}}[]
    lower,upperbox=box
    for (k,(fl,fu)) in enumerate(floats)
        l,u=Exact.(fl),Exact.(fu); fails=0; refined=0
        for j in eachindex(l)
            a,b=affineinterval(layers[k],lower,upperbox,j)
            if b>u[j]
                c=zeros(Exact,length(l));c[j]=1
                b=min(b,upper(layers,k,c,hidden,box));refined+=1
            end
            if a<l[j]
                c=zeros(Exact,length(l));c[j]=-1
                a=max(a,-upper(layers,k,c,hidden,box));refined+=1
            end
            if a<l[j] || b>u[j]
                fails+=1
                fails<=3 && println("Unproved ",i," ",k," ",j," lower gap ",Float64(l[j]-a)," upper gap ",Float64(b-u[j]))
            end
        end
        println("image=",i," layer=",k," intervals=",length(l)," refined_sides=",refined," unproved=",fails);flush(stdout)
        @assert fails==0
        frac=[a<0<b ? Exact(clamp(Float64(b/(b-a)),0.,1.)) : zero(Exact) for (a,b) in zip(l,u)]
        push!(hidden,(l,u,frac))
        lower,upperbox=max.(l,0),max.(u,0)
    end
    for j in setdiff(eachindex(layers[end].b),y)
        c=zeros(Exact,length(layers[end].b));c[y]=-1;c[j]=1
        exact_bound=upper(layers,length(layers),c,hidden,box)
        float_bound=margin_upper(stages,floats,lo,hi,y,j)
        @assert exact_bound<=Exact(float_bound) (i,j,Float64(exact_bound),float_bound)
        println("digit=",y-1," target=",j-1," upper=",Float64(exact_bound))
    end
end
indices = isempty(ARGS) ? [row.sample_index for row in CSV.File(joinpath(@__DIR__, "research-fallback-images.csv"))] : parse.(Int, ARGS)
for image in indices
    audit(image)
end
