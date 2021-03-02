module Inference

using GZip
using LinearAlgebra, Statistics, StatsBase

include("io.jl")
using .DataIO

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
    return [y[k] for k in index]
end

# ------------------------------------------------------------------------
# main functions

function cost(ref, qry; α=1, β=1, γ=0, ω=nothing)
    ϕ = match(ref.gene, qry.gene)

    Σ = zeros(size(ref.data,1), size(qry.data,1))
    for i in 1:size(ref.data,2)
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

    return Matrix(Σ)
end

function sinkhorn(M::Array{Float64,2};
                  a::Maybe{Array{Float64}} = missing, 
                  b::Maybe{Array{Float64}} = missing,
                  maxᵢ::Integer            = 1000,
                  τ::Real                  = 1e-5,
                  verbose::Bool            = false)
    i = 0
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

    Σ = cost(ref, qry; α=1.0, β=2.6, γ=0.65) # TODO: expose parameters?

    return (
        invert=(β) -> sinkhorn(exp.(-(1 .+ β*Σ))),
        pointcloud = pointcloud,
    )
end

end
