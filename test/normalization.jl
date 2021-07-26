module Normalize

using LinearAlgebra
using Statistics, StatsBase
using Random, Distributions

include("../src/util.jl")
using .Utility

# XXX: return coefficients (?)
function random_polynomial(order, scale)
    a = randn(order+1, order+1)
    return (u,v) -> sum(scale^(p₁+p₂)*a[p₁+1,p₂+1]*(u-.5)^p₁*(v-.5)^p₂ for p₁ ∈ 0:order for p₂ ∈ 0:order if (p₁ + p₂) ≤ order)
end

function rescale(x)
    x = x .- minimum(x,dims=2)
    x = x ./ mean(x,dims=2)

    return x
end

function random_surface(dim, len, order, scale)
    u, v = rand(len), rand(len)
    Λ = [ random_polynomial(order, scale) for d ∈ 1:dim ]
    w = hcat((λ.(u,v) for λ ∈ Λ)...)'
    w = rescale(w)

    return w, u, v
end

function random_mean(w, ngene)
    ncell = size(w,2)

    g = rand(Gamma(5,5), ngene) .+ 1
    c = rand(Gamma(5,5), ncell) .+ 1
    μ = Diagonal(g)*(rand(ngene,size(w,1))/size(w,1))*w*Diagonal(c)

    return μ, (gene=g, cell=c)
end

function random_sample(μ)
    ngene, ncell = size(μ)

    ϕ = rand(Gamma(10,10), ngene) .+ 1
    sample = Array{Int,2}(undef, ngene, ncell)
    for i in 1:ngene
        r = ϕ[i]
        for j in 1:ncell
            p = ϕ[i] / (μ[i,j] .+ ϕ[i])
            sample[i,j] = rand(NegativeBinomial(r,p))
        end
    end

    return sample, ϕ
end

function generate(ngene, ncell, rank; order=10)
    w, u, v   = random_surface(rank, ncell, order, 2)
    μ, scale  = random_mean(w, ngene)
    sample, ϕ = random_sample(μ)

    return (
        sample = sample,
        mean   = μ,
        ϕ      = ϕ,
        scale  = scale,
        w      = w,
        u      = u,
        v      = v,
    )
end

function main()
    model = generate(8000,2000,25; order=15)
    σ² = model.sample.*(model.sample .+ model.ϕ) ./ (1 .+ model.ϕ)
    u, v = sinkhorn(σ²)

    N = Diagonal(sqrt.(u))*model.sample*Diagonal(sqrt.(v))
    F = svd(N)

    return model, (norm=N, svd=F), u, v
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    Normalize.main()
end
