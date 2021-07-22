module Normalize

using LinearAlgebra
using Statistics, StatsBase
using Random, Distributions

include("../src/util.jl")
using .Utility

function generate(ngene, ncell, rank)
    W, H = 2*rand(ngene,rank)/rank, 2*rand(rank,ncell)/rank

    ϕ = rand(Exponential(10), ngene) .+ 1
    g = rand(Gamma(10,2), ngene) .+ 1
    c = rand(Gamma(10,2), ncell) .+ 1
    μ = Diagonal(g)*W*H*Diagonal(c)

    N = Array{Int,2}(undef, ngene, ncell)
    for i in 1:ngene
        r = ϕ[i]
        for j in 1:ncell
            p = ϕ[i] / (μ[i,j] .+ ϕ[i])
            N[i,j] = rand(NegativeBinomial(r,p))
            # N[i,j] = rand(Poisson(μ[i,j]))
        end
    end

    return (
        mean=μ,
        cell=c,
        gene=g,
        factor=ϕ,
        sample=N
    )
end

function main()
    model = generate(2000,1000,25)
    σ² = model.sample.*(model.sample .+ model.factor) ./ (1 .+ model.factor)
    u, v = sinkhorn(σ²; verbose=true)
    σ²₁ = Diagonal(u)*σ²*Diagonal(v)

    x, y = sinkhorn(model.sample, model.factor; verbose=true)
    N   = Diagonal(x)*model.sample*Diagonal(y)
    σ²₂ = N.*(N.+ model.factor) ./ (1 .+ model.factor)

    return model, u, v, σ²₁, x, y, σ²₂
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    Normalize.main()
end
