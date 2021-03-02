module ML

using Random, Statistics
using LinearAlgebra, Flux, Zygote

import Base:
    length, reverse, iterate

export model, train!, validate, preprocess, update_dimension

# NOTE: all data is assumed to be dᵢ x N shaped here!
#       this is to allow for fast matrix multiplication through the network
#       it might be worthwhile to keep both memory layouts stored...

# ------------------------------------------------------------------------
# globals

σ₀(x) = x + (sqrt(x^2 +1)-1)/2

# ------------------------------------------------------------------------
# types

# Iterator
struct LayerIterator
    width     :: Array{Int}
    dropout   :: Set{Int}
    normalize :: Set{Int}
    σᵢₒ       :: Function # activation on input/output layers
    σ         :: Function # activation on latent layers
end

length(it::LayerIterator)  = length(it.width) + length(it.dropout) + length(it.normalize)
reverse(it::LayerIterator) = LayerIterator(
                                reverse(it.width),
                                Set(length(it.width) - i - 1 for i in it.dropout),
                                Set(length(it.width) - i - 1 for i in it.normalize),
                                it.σᵢₒ,
                                it.σ,
                             )

function iterate(it::LayerIterator)
    w₁ = it.width[1]
    w₂ = it.width[2]
    f  = Dense(w₁, w₂, it.σᵢₒ) |> gpu

    return f, (
        index     = 2,
        dropout   = 1 ∈ it.dropout,
        normalize = 1 ∈ it.normalize,
    )
end

function iterate(it::LayerIterator, state)
    return if state.dropout
               Dropout(0.5) |> gpu, (
                   index     = state.index,
                   dropout   = false,
                   normalize = state.normalize,
               )
           elseif state.normalize
               BatchNorm(it.width[state.index]) |> gpu, (
                   index     = state.index,
                   dropout   = false,
                   normalize = false,
               )
           elseif state.index < length(it.width)
                w₁ = it.width[state.index]
                w₂ = it.width[state.index+1]

                i  = state.index+1
                f  = Dense(w₁, w₂, i == length(it.width) ? it.σᵢₒ : it.σ) |> gpu

                f, (
                     index     = i,
                     dropout   = (i-1) ∈ it.dropout,
                     normalize = (i-1) ∈ it.normalize,
                )
           else
               nothing
           end
end

# ------------------------------------------------------------------------
# data scaling / preprocessing

# x is assumed to be dᵢ x N shaped
function preprocess(x; dₒ::Union{Nothing,Int}=nothing, ϕ=(x)->x)
    X = gpu(x)
	F = svd(X)
	
	d = F.Vt
	μ = mean(d, dims=2)
	σ = std(d, dims=2)
	
    λ = ϕ.(F.S)
    λ = λ ./ sum(λ)
	
    if isnothing(dₒ)
        dₒ = size(d,1)
    end

	return (
        data   = (d[1:dₒ,:] .- μ[1:dₒ]) ./ σ[1:dₒ], 
        weight = λ[1:dₒ], 
        map    = (x) -> (F.U[:,1:dₒ] * Diagonal(F.S[1:dₒ])) * ((σ[1:dₒ]) .* x .+ μ[1:dₒ])
    )
end

# ------------------------------------------------------------------------
# functions

function model(dᵢ, dₒ; Ws=Int[], normalizes=Int[], dropouts=Int[], σ=elu)
    # check for obvious errors here
    length(dropouts) > 0   && length(Ws) < maximum(dropouts) ≤ 0   && error("invalid dropout layer position")
    length(normalizes) > 0 && length(Ws) < maximum(normalizes) ≤ 0 && error("invalid normalization layer position")

    layers = LayerIterator(
                    [dᵢ; Ws; dₒ], 
                    Set(dropouts),
                    Set(normalizes), 
                    σ₀, σ
             )

    F   = Chain(layers...)
    F¯¹ = Chain(reverse(layers)...)
    𝕀   = Chain(F, F¯¹)

    return (
        pullback=F,
        pushforward=F¯¹,
        identity=𝕀
    )
end

function update_dimension(model, dₒ; ϵ = 1e-6)
    F, F¯¹, 𝕀 = model

    lᵢ = F[end]
    lₒ = F¯¹[1]

    Wᵢ, bᵢ = params(lᵢ)
    Wₒ, bₒ = params(lₒ)

    size(Wᵢ,1) == dₒ && return nothing
    size(Wᵢ,1) >  dₒ && error("can not reduce dimensionality of model") 

    δ  = dₒ - size(Wᵢ, 1)

    W̄ᵢ = vcat(Wᵢ, ϵ*randn(δ, size(Wᵢ,2)))
    b̄ᵢ = vcat(bᵢ, ϵ*randn(δ))
    l̄ᵢ = Dense(W̄ᵢ, b̄ᵢ, lᵢ.σ)

    W̄ₒ = hcat(Wₒ, ϵ*randn(size(Wₒ,1), δ))
    b̄ₒ = bₒ
    l̄ₒ = Dense(W̄ₒ, b̄ₒ, lₒ.σ)

    F   = Chain( (i < length(F) ? f : l̄ᵢ for (i,f) ∈ enumerate(F))...)
    F¯¹ = Chain( (i > 1 ? f : l̄ₒ for (i,f) ∈ enumerate(F¯¹))...)
    𝕀   = Chain(F, F¯¹)

    return (
        pullback=F,
        pushforward=F¯¹,
        identity=𝕀
    )
end

# loss function factories
reconstruction_loss(model, Ω) = (x) -> begin
    x̂ = model.identity(x)
    return sum(ω*mse(x[i,:], x̂[i,:]) for (i,ω) in enumerate(Ω))
end

# data batching
function batch(data, n)
    N = size(data,2)

    lo(i) = (i-1)*n + 1
    hi(i) = min((i)*n, N)

    ι = randperm(N)

    return (data[:,ι[lo(i):hi(i)]] for i in 1:ceil(Int, N/n)), 
           (ι[lo(i):hi(i)] for i in 1:ceil(Int, N/n))
end

function validate(data, len)
    ι = randperm(size(data,2))
    return (
        valid = data[:,ι[1:len]],
        train = data[:,ι[len+1:end]],
    ),(
        valid = ι[1:len],
        train = ι[len+1:end],
    )
end

function noop(epoch) end

# data training
function train!(model, data, loss; B=64, η=1e-3, N=100, log=noop)
    Θ   = params(model.identity)
    opt = ADAM(η)

    trainmode!(model.identity)
    for n ∈ 1:N
        X, I = batch(data, B)
        for (i,x) ∈ zip(I,X)
            E, backpropagate = pullback(Θ) do
                loss(x, i, false)
            end

            isnan(E) && @goto done

            ∇Θ = backpropagate(1f0)
            Flux.Optimise.update!(opt, Θ, ∇Θ)
        end

        log(n)
    end
    @label done
    testmode!(model.identity)
end

# ------------------------------------------------------------------------
# tests

end
