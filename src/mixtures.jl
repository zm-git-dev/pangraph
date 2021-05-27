module Mixtures

using Roots, SpecialFunctions, StatsFuns
using Random, Distributions, Statistics

export fitmixture, EM

function MLE(x; γ=missing) 
    if ismissing(γ)
        γ = ones(size(x))
    end
    N = sum(γ)
    μ = sum((γ.*x))/N
    σ = log(μ) - sum(γ.*log.(x))/N

    # initial guess
    α = ((3 - σ) + sqrt((σ-3)^2 + 24σ)) / (12σ)
    if isinf(α)
        α = 10
    end

    try
        F(α) = log(α) - digamma(α) - σ
        α = find_zero(F, α, Order1())
    catch
        println("ERROR:")
        println("bounds=($(minimum(x)), $(maximum(x)))")
        println("N=$(N)")
        println("σ=$(σ)")
        println("μ=$(μ)")
        println(((3 - σ) + sqrt((σ-3)^2 + 24σ)) / (12σ))
    end
    β = α/μ

    return α, β
end

function weights(γ)
    return sum(γ, dims=1)'/sum(γ)
end

logρ(x,α,β) = (α*log(β) - loggamma(α)) .+ (α-1).*log.(x) .- β.*x 
ρ(x,α,β)    = (x != 0) ? exp(logρ(x,α,β)) : β^α / gamma(α)

function posterior(x, Π, α, β; dropout=false)
    γ = Array{Float64, 2}(undef, length(x), length(α))

    for k = 1:length(α)
        γ[:,k] = Π[k]*ρ.(x, α[k], β[k])
    end

    if dropout
        χ = minimum(x[x.>0])
        p = vec([gammacdf(α[k], 1/β[k], χ) for k in 1:length(α)])
        γ[findall(x.==0),:] .= p'
    end

    if any(isinf.(γ))
        γ[isinf.(γ)] .= 1
    end

    γ = γ ./ sum(γ, dims=2)

    return γ
end

loglikelihood(x, Π, α, β) = sum(log.(sum(Π' .* hcat([ρ.(x,α[k],β[k]) for k in 1:length(α)]...), dims=2)))

function sample(Π, α, β, N; k=missing)
    if ismissing(k)
        MM = MixtureModel(Gamma, [(α[k], 1/β[k]) for k in 1:length(α)], vec(Π))
    else
        MM = Gamma(α[k], 1/β[k])
    end
    return rand(MM, N)
end

function kmeans(x, k::Integer; verbose = false, maxᵢ::Integer = 1000)
    q  = [quantile(x, q) for q in range(0, 1, length=k+1)]
    μ  = [mean(x[(q[i-1] .<= x) .& (x .<= q[i])]) for i in 2:length(q)]
    S₀ = zeros(length(x), 1)
    for n in 1:maxᵢ
        S = getindex.(argmin(abs.(x .- μ'), dims=2), 2)
        if !any(x->x!=0, S-S₀)
            break
        end
        if verbose
            println("iteration $(n): μ = $(μ)")
        end
        μ  = [mean(x[findall(S.==i)]) for i in 1:k]
        S₀ = S
    end

    σ = [std(x[findall(S₀.==i)]) for i in 1:k]

    return μ, σ, S₀
end

function EM(x, K::Integer; ϵ::Real = 1e-5, maxᵢ::Integer = 10000, verbose::Bool = false)
    μ, σ, S₀ = kmeans(x, K)

    β = μ./σ.^2 
    α = μ.*β
    Π = [sum(S₀.==k)/length(S₀) for k in 1:K]

    δ  = Inf
    L₀ = loglikelihood(x, Π, α, β)
    i  = 0
    while abs(δ) > ϵ && i < maxᵢ
        # expectation + maximization
        γ = posterior(x, Π, α, β)

        Π = weights(γ)
        Θ = [MLE(x, γ=γ[:,k]) for k in 1:K]
        α = [θ[1] for θ in Θ]
        β = [θ[2] for θ in Θ]

        L = loglikelihood(x, Π, α, β)
        if verbose
            println("iteration $(i). loglikelihood = $(L)")
        end
        δ  = L - L₀
        L₀ = L
        i += 1
    end

    ι = sortperm(α./β)

    return L₀, (Π[ι], α[ι], β[ι])
end

function cluster(x, Π, α, β; dropout=false)
    println("computing posterior")
    γ = posterior(x, Π, α, β, dropout=dropout)
    println("getting index")
    χ = getindex.(argmax(γ, dims=2), 2)
    println("returning")
    return [x[findall(χ .== k)] for k in 1:length(α)]
end

# XXX: right now just returns the posteriors for component with largest mean
function fitmixture(data)
    _, params = EM(data.+1e-6, 2)
    ρ = posterior(data, params...)
    return ρ[:,2]
end

end
