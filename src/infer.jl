module Inference

using GZip
using LinearAlgebra, Statistics, StatsBase

include("io.jl")
using .DataIO

include("mixtures.jl")
using .Mixtures

# ------------------------------------------------------------------------
# exports

export virtualembryo

# ------------------------------------------------------------------------
# globals

Maybe{T} = Union{T, Missing}

# ------------------------------------------------------------------------
# helper functions

function cumulative(data)
    d = sort(data)
    function F(x)
        i = searchsortedfirst(d, x)
        return (i-1)/length(d)
    end

    return F
end

columns(genes) = Dict(g=>i for (i,g) ∈ enumerate(genes))

function virtualembryo()
    expression, _, genes = GZip.open("$root/dvex/bdtnp.txt.gz") do io
        read_matrix(io; named_cols=true)
    end

    positions, _, _, = GZip.open("$root/dvex/geometry_reduced.txt.gz") do io
        read_matrix(io; named_cols=true)
    end

    return (
        expression = (
            data = expression,
            gene = columns(genes),
        ),
        position  = positions
    )
end

function scrna()
    expression, genes, _ = GZip.open("$root/dvex/dge_normalized.txt.gz") do io
        read_matrix(io; named_cols=true, named_rows=true)
    end

    return (
        data = expression',
        gene = columns(genes),
    )
end

function match(x, y)
    index = Array{String}(undef,length(x))
    for (g,i) in x
        index[i] = g
    end
    return [k ∈ keys(y) ? y[k] : nothing for k in index]
end

# ------------------------------------------------------------------------
# main functions

function cost(ref, qry; α=1, β=1, γ=0, ω=nothing)
    ϕ = match(ref.gene, qry.gene)

    Σ = zeros(size(ref.data,1), size(qry.data,1))
    for i in 1:size(ref.data,2)
        isnothing(ϕ[i]) && continue

        r  = ref.data[:,i]
        q  = qry.data[:,ϕ[i]]
        ω₀ = isnothing(ω) ? 1 : ω[i]

        f = sum(q .== 0) / length(q)
        χ = quantile(r, f)

        F₀ = cumulative(r[r.<=χ])
        F₊ = cumulative(r[r.>χ])
        F₌ = cumulative(q[q.>0])

        for j in 1:size(ref.data,1)
            for k in 1:size(qry.data,1)
                if r[j] > χ && q[k] > 0
                    Σ[j,k] += -ω₀*(2*F₊(r[j])-1)*(2*F₌(q[k])-1)
                elseif r[j] > χ && q[k] == 0
                    Σ[j,k] += +ω₀*(α*F₊(r[j])+γ)
                elseif r[j] <= χ && q[k] > 0
                    Σ[j,k] += +ω₀*(α*F₌(q[k])+γ)
                else
                    Σ[j,k] += +ω₀*β*F₀(r[j])
                end
            end
        end
    end

    return Matrix(Σ), ϕ
end

function cost_simple(ref, qry)
    ϕ = match(ref.gene, qry.gene)
    Σ = zeros(size(ref.data,1), size(qry.data,1))

    @show sum(1 for elt ∈ ϕ if !isnothing(elt))

    σ(x) = x
    for i in 1:size(ref.data,2)
        isnothing(ϕ[i]) && continue

        r  = ref.data[:,i]
        q  = qry.data[:,ϕ[i]]

        R = 2*σ.((rank(r)./length(r))) .- 1
        Q = 2*σ.((rank(q)./length(q))) .- 1

        Σ -= (R*Q')

        #=
        for j in 1:size(ref.data,1)
            for k in 1:size(qry.data,1)
                Σ[j,k] += -*(2*σ(R(r[j]).^4)-1)*(2*σ(Q(q[k]).^4)-1)
            end
        end
        =#
    end

    return Matrix(Σ), ϕ
end

rank(x) = invperm(sortperm(x))

function cost_scan(ref, qry, ν, ω)
    ϕ = match(ref.gene, qry.gene)
    Σ = zeros(size(ref.data,1), size(qry.data,1))

    # σ(x) = x
    k    = 1
    σ(x) = 1 ./ (1+((1-x)/x)^k)

    for i in 1:size(ref.data,2)
        isnothing(ϕ[i]) && continue

        r  = ref.data[:,i]
        q  = qry.data[:,ϕ[i]]

        R = 2*σ.((rank(r)./length(r)).^ν[i]) .- 1
        Q = 2*(rank(q)./length(q)) .- 1

        Σ -= ω[i]*(R*Q')
    end

    return Matrix(Σ), ϕ
end

function sinkhorn(M::Array{Float64,2};
                  a::Maybe{Array{Float64}} = missing, 
                  b::Maybe{Array{Float64}} = missing,
                  maxᵢ::Integer            = 1000,
                  τ::Real                  = 1e-5,
                  verbose::Bool            = false)
    c = 1 ./sum(M, dims=1)
    r = 1 ./(M*c')

    if ismissing(a)
        a = ones(size(M,1), 1) ./ size(M,1)
    end
    if ismissing(b)
        b = ones(size(M,2), 1) ./ size(M,2)
    end

    if length(a) != size(M,1)
        throw(error("invalid size for row prior"))
    end
    if length(b) != size(M,2)
        throw(error("invalid size for column prior"))
    end

    i = 0
    rdel, cdel = Inf, Inf
    while i < maxᵢ && (rdel > τ || cdel > τ)
        i += 1

        cinv = M'*r
        cdel = maximum(abs.(cinv.*c .- b))
        c    = b./cinv

        rinv = M*c
        rdel = maximum(abs.(rinv.*r .- a))
        r    = a./rinv

        if verbose
            println("Iteration $i. Row = $rdel, Col = $cdel")
        end

    end

    if verbose
        println("Terminating at iteration $i. Row = $rdel, Col = $cdel")
    end

    return M.*(r*c')
end

function inversion()
    ref, pointcloud = virtualembryo()
    qry = scrna()

    Σ, _ = cost(ref, qry; α=1.0, β=2.6, γ=0.65) # TODO: expose parameters?

    return (
        invert=(β) -> sinkhorn(exp.(-(1 .+ β*Σ))),
        pointcloud = pointcloud,
    )
end

function inversion(counts, genes; ν=nothing, ω=nothing)
    ref, pointcloud = virtualembryo()
    qry = (
        data=counts', 
        gene=columns(genes),
    )

    Σ, match = 
        if isnothing(ν) || isnothing(ω)
            cost_simple(ref, qry)
        else
            cost_scan(ref, qry, ν, ω)
        end

    return (
        invert     = (β) -> sinkhorn(exp.(-(1 .+ β*Σ))),
        pointcloud = pointcloud,
        database   = (
            data=ref.data',
            gene=ref.gene,
        ),
        match      = match
    )
end

# basic statistics
mean(x)  = sum(x) / length(x)
cov(x,y) = mean(x.*y) .- mean(x)*mean(y)
var(x)   = cov(x,x)
std(x)   = sqrt(abs(var(x)))
cor(x,y) = cov(x,y) / (std(x) * std(y))

function make_objective(ref, qry)
    function objective(Θ)
        β, ν, ω = 0.1, Θ[1:84], Θ[85:end]
        Σ, ϕ    = cost_scan(ref, qry, ν, ω)

        ψ = sinkhorn(exp.(-(1 .+ β*Σ)))
        ψ *= minimum(size(ψ))

        ι   = findall(.!isnothing.(ϕ))
        db  = ref.data[:,ι]
        est = ψ*qry.data[:,ϕ[ι]]

        return 1-mean(cor(db[:,i], est[:,i]) for i in 1:size(db,2))
    end

    return objective
end

# using Optim
using BlackBoxOptim

function scan_params(count, genes)
    qry = (
        data=count',
        gene=columns(genes),
    )
    ref, _ = virtualembryo()

    f = make_objective(ref,qry)

    # return optimize(f, zeros(1+84), fill(100.0, 1+84), [0.1; ones(84)], SAMIN(;rt=0.9,verbosity=1), 
    #             Optim.Options(
    #                 show_trace=true,
    #                 show_every=5,
    #                 iterations=200,
    #             )
    # )

    return bboptimize(f, 
                      SearchRange=[[(0.01, 100.0) for _ ∈ 1:84]; [(0.01, 100.0) for _ ∈ 1:84]], 
                      MaxFuncEvals=5000,
                      Method=:generating_set_search,
                      TraceMode=:compact
    ) #, Method=:dxnes, NThreads=Threads.nthreads(), )
end

end
