module Utility

using LinearAlgebra
using Statistics, StatsBase

export sinkhorn

function sinkhorn(A; r=[], c=[], maxit=1000, δ=1e-6, verbose=false)
    if length(r) == 0
        r = size(A,2)
    end
    
    if length(c) == 0
        c = size(A,1)
    end
    
    x = ones(size(A,1))
    y = ones(size(A,2))
    for i ∈ 1:maxit
        δr = maximum(abs.(x.*(A*y)  .- r))
        δc = maximum(abs.(y.*(A'*x) .- c))
        
        if verbose
            @show minimum(A'*x), minimum(A*y)
            @show r, c
            @show i, δr, δc
        end
        
        (isnan(δr) || isnan(δc)) && return x, y, false
        (δr < δ && δc < δ) 		 && return x, y, true
        
        y = c ./ (A'*x)
        x = r ./ (A*y)
        if verbose
            @show mean(x), mean(y)
        end
    end
    
    return x, y, false
end

function sinkhorn(N, ϕ; maxit=1000, δ=1e-6, verbose=false)
    nrow = size(N,1)
    ncol = size(N,2)
    
    N² = N.^2
    u, v = ones(size(N,1)), ones(size(N,2))
    for i ∈ 1:maxit
        σ² = ((Diagonal(u.^2 ./ (1 .+ ϕ))*N²*Diagonal(v.^2)) .+ (Diagonal(u.*ϕ ./ (1 .+ ϕ))*N*Diagonal(v)))
        δr = maximum(abs.(sum(σ², dims=1) .- nrow))
        δc = maximum(abs.(sum(σ², dims=2) .- ncol))

        if verbose
            @show i, δr, δc
        end

        (isnan(δr) || isnan(δc)) && return u, v, false
        (δr < δ && δc < δ) 		 && return u, v, true

        A = vec(sum(Diagonal(u.^2 ./ (1 .+ ϕ)) * N², dims=1))
        B = vec(sum(Diagonal(u.*ϕ ./ (1 .+ ϕ)) * N , dims=1))

        v = @. (sqrt(B^2 + 4*A*nrow) - B) / (2*A)

        A = vec(sum(N² * Diagonal(v.^2),           dims=2))
        B = vec(sum(Diagonal(ϕ) * N * Diagonal(v), dims=2))

        u = @. (sqrt(B^2 + 4*A*ncol*(1+ϕ)) - B) / (2*A)
    end
    
    return u, v, false
end

end
