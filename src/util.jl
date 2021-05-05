module Utility

using LinearAlgebra
using Statistics, StatsBase

function sinkhorn(A; r=[], c=[], maxᵢₜ=1000, δ=1e-4, verbose=false)
    if length(r) == 0
        r = size(A,2)
    end
    
    if length(c) == 0
        c = size(A,1)
    end
    
    x = ones(size(A,1))
    y = ones(size(A,2))
    for i ∈ 1:maxᵢₜ
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

end
