module Distances

using LinearAlgebra
using Statistics, StatsBase

export euclidean

function euclidean(X)
    dotprod = X'*X
    vecnorm = vec(diag(dotprod))
    
    return .√((vecnorm' .+ vecnorm) .- 2*dotprod)
end

kullback_liebler(p, q) = sum(x .* log(x ./ y) for (x,y) ∈ zip(p,q) if x > 0 && y > 0)

function jensen_shannon(P)
    D = zeros(size(P,2), size(P,2))
    Threads.@threads for i in 1:(size(P,2)-1)
        for j in (i+1):size(P,2)
            M = (P[:,i] + P[:,j]) / 2
            D[i,j] = (kullback_liebler(P[:,i],M) + kullback_liebler(P[:,j],M)) / 2
            D[j,i] = D[i,j]
        end
    end

    D = .√D
    return D / mean(D[:])
end

end
