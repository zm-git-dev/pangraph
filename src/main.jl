module SeqSpace

using GZip
using BSON: @save
using LinearAlgebra: norm
using Statistics: quantile
using Flux, Zygote

import BSON

include("geo.jl")
include("io.jl")
include("rank.jl")
include("model.jl")
include("hilbert.jl")
include("voronoi.jl")

using .PointCloud, .DataIO, .SoftRank, .ML

export Result, HyperParams
export run

# ------------------------------------------------------------------------
# globals

# ------------------------------------------------------------------------
# types

mutable struct HyperParams
    dₒ :: Int          # output dimensionality
    Ws :: Array{Int,1} # (latent) layer widths
    BN :: Array{Int,1} # (latent) layers followed by batch normalization
    DO :: Array{Int,1} # (latent) layers followed by drop outs
    N  :: Int          # number of epochs to run
    δ  :: Int          # epoch subsample factor for logging
    η  :: Float64      # learning rate
    B  :: Int          # batch size
    V  :: Int          # number of points to partition for validation
    k :: Int           # number of neighbors to use to estimate geodesics
    γₓ :: Float64      # prefactor of neighborhood isometry loss
    γₛ :: Float64      # prefactor of distance soft rank loss
    γₙ :: Float64      # prefactor 
end

HyperParams(; dₒ=3, Ws=Int[], BN=Int[], DO=Int[], N=500, δ=5, η=5e-4, B=64, V=81, kₙ=12, kₗ=3, γₓ=1e-4, γₛ=1, γₙ=5000) = HyperParams(dₒ, Ws, BN, DO, N, δ, η, B, V, kₙ, kₗ, γₓ, γₛ, γₙ)

struct Result
    param :: HyperParams
    loss  :: NamedTuple{(:train, :valid), Tuple{Array{Float64,1},Array{Float64,1}} }
    model
end

# ------------------------------------------------------------------------
# utility functions

# α := rescale data by
# δ := subsample data by
function pointcloud(;α=1, δ=1)
    verts, _ = open("$root/gut/mesh_apical_stab_000153.ply") do io
        read_ply(io)
    end

    return α*vcat(
        map(v->v.x, verts)',
        map(v->v.y, verts)',
        map(v->v.z, verts)'
    )[:,1:δ:end]
end

function expression(;raw=false)
    scrna, genes, _  = if raw
        GZip.open("$root/dvex/dge_raw.txt.gz") do io
            read_matrix(io; named_cols=false, named_rows=true)
        end
    else
        GZip.open("$root/dvex/dge_normalized.txt.gz") do io
            read_matrix(io; named_cols=true, named_rows=true)
        end
    end

    return scrna, genes
end

mean(x)    = sum(x) / length(x)
median(x)  = sort(x)[length(x)÷2+1]
var(x)     = mean((x.-mean(x)).^2)
std(x)     = sqrt(var(x))
cov(x,y)   = mean((x.-mean(x)) .* (y.-mean(y)))

ball(D, k) = [ sort(view(D,:,i))[k+1] for i in 1:size(D,2) ]

# ------------------------------------------------------------------------
# i/o

# TODO: add a save function

# assumes a BSON i/o
function load(io)
    database = BSON.parse(io)
    result, input = database[:result], database[:in]
end

function buildloss(model, D², param)
    clamp(x) = if x > 1
        1
    elseif x < 0
        0
    else
        x
    end

    (x, i, log) -> begin
        z = model.pullback(x)
        x̂ = model.pushforward(z)

        # reconstruction loss
        ϵᵣ = sum(sum((x.-x̂).^2, dims=2)) / sum(sum(x.^2,dims=2))

        # distance softranks
        D̂² = distance²(z)
        D̄² = D²[i,i]
        
        ϵₛ = mean(
            let
                d, d̂ = D̄²[:,j], D̂²[:,j]
                r, r̂ = softrank(d ./ mean(d)), softrank(d̂ ./ mean(d̂))
                1 - cov(r, r̂)/(std(r)*std(r̂))
            end for j ∈ 1:size(D̂²,2)
        )

        # FIXME: allow for higher dimensionality than 2
        ϵᵤ = let
            a = Voronoi.areas(z[1:2,:])
            std(a) / mean(a) + mean(
                # FIXME: remove when we can compute volumes in d > 2
                let
                    zₛ = sort(z[d,:])
                    mean( ( (2*i/length(zₛ)-1) - s)^2 for (i,s) in enumerate(zₛ) )
                end for d ∈ 3:size(z,1)
            )
        end

        if log
            @show ϵᵣ, ϵₛ, ϵᵤ
        end

        return ϵᵣ + param.γₛ*ϵₛ + param.γᵤ*ϵᵤ
     end
end

# ------------------------------------------------------------------------
# main functions

function linearprojection(x, d; Δ=1, Λ=nothing) 
    Λ = isnothing(Λ) ? svd(x) : Λ

	ψ = Λ.Vt
	μ = mean(ψ, dims=2)
    λ = Λ.S ./ sum(Λ.S)
    ι = (1:d) .+ Δ

    x₀ = F.U[:,1:Δ]*Diagonal(F.S[1:Δ])*F.Vt[1:Δ,:]
	return (
        projection = ψ[ι,:] .- μ[ι] 
        embed = (x) -> (x₀ .+ (F.U[:,ι]*Diagonal(F.S[ι]))*(x.+μ[ι]))
    )
end

function run(data, param; D²=nothing)
    D² = isnothing(D²) ? geodesics(x⃗, param.k).^2 : D²

    M = model(size(x⃗, 1), param.dₒ; 
          Ws         = param.Ws, 
          normalizes = param.BN, 
          dropouts   = param.DO
    )

    y⃗, I = validate(x⃗, param.V)

    R          = ball(D², param.k)
    vecnorm(x) = sum(norm.(eachcol(x)))

    loss = buildloss(M, D², param)
    E    = (
        train = Float64[],
        valid = Float64[]
    )

    log = (n) -> begin
        if (n-1) % param.δ == 0 
            @show n

            push!(E.train,loss(y⃗.train, I.train, true))
            push!(E.valid,loss(y⃗.valid, I.valid, true))
        end

        nothing
    end

    train!(M, y⃗.train, I.train, loss; 
        η   = param.η, 
        B   = param.B, 
        N   = param.N, 
        log = log
    )

    return Result(param, E, M), (x=x⃗, map=ϕ, y=y⃗, index=I, D²=D², log=log)
end

function extendrun(result::Result, input, epochs)
    loss = buildloss(result.model, input.D², result.param)
    train!(result.model, input.y.train, input.index.train, loss; 
        η   = result.param.η, 
        B   = result.param.B, 
        N   = epochs,
        log = input.log
    )

    return result, input
end

function main(input, niter, output)
    params = Result[]
    open(input, "r") do io 
        params = eval(Meta.parse(read(io, String)))
    end
    
    result, data = run(params, niter)
    @save "$root/result/$output.bson" result data
end

end
