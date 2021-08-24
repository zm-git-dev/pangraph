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
    kₙ :: Int          # number of neighbors to use to estimate geodesics
    kₗ :: Int          # average number of neighbors to impose isometry on 
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

# ball(D, k) = mean(sort(view(D,:,i))[k+1] for i in 1:size(D,2))
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
    #=
    # Lₛ = 20
    # xₛ = Lₛ*(rand(param.dₒ,7500) .- .5)
    # Dₛ = sort(PointCloud.upper_tri(PointCloud.distance(xₛ)))
    # dq = Dict{Int,Vector}()
    # function distance_quantiles(N)
    #     if N ∉ keys(dq)
    #         dq[N] = [quantile(Dₛ, i/N; sorted=true) for i in 1:N]
    #     end
    #     return dq[N]
    # end
    
    N  = 10
    Δs = range(-1,+1,length=N)
    Δ  = [((Δs[i],Δs[i+1]), (Δs[j],Δs[j+1])) for i in 1:(N-1), j in 1:(N-1)]
    
    rect = Rectangle(Point2(-1,-1), Point2(+1,+1))
    =#

    (x, i, log) -> begin
        z = model.pullback(x)
        x̂ = model.pushforward(z)

        # reconstruction loss
        ϵᵣ = sum(sum((x.-x̂).^2, dims=2)) / sum(sum(x.^2,dims=2))

        # pairwise distances
        D̂² = distance²(z)
        D̄² = D²[i,i]
        
        # d = upper_tri(D̄²)
        # d̂ = upper_tri(D̂²)

        #=
        ϵₓ = mean(
            let
                zₛ = sort(z[d,:])
                mean( ( (2*i/length(zₛ)-1) - s)^2 for (i,s) in enumerate(zₛ) )
            end for d ∈ 1:min(2,size(z,1))
        )
        =#
        # ϵₓ = let
        #     zₛ = Hilbert.sort(z)
        #     z₀ = Zygote.dropgrad(Hilbert.positions(range(0,1,length=size(z,2))))
        #     (mean(sum((zₛ .- z₀).^2, dims=1)) + mean(
        #         let
        #             zₛ = sort(z[d,:])
        #             mean( ( (2*i/length(zₛ)-1) - s)^2 for (i,s) in enumerate(zₛ) )
        #         end for d ∈ 1:min(2,size(z,1))
        #    ))
        # end
        # FIXME: allow for higher dimensionality than 2
        ϵₓ = let
            a = Voronoi.areas(z[1:2,:])
            # mean((length(a)*a/4 .- 1).^2)
            std(a) / mean(a) + mean(
                # FIXME: remove when we can compute volumes in d > 2
                let
                    zₛ = sort(z[d,:])
                    mean( ( (2*i/length(zₛ)-1) - s)^2 for (i,s) in enumerate(zₛ) )
                end for d ∈ 3:size(z,1)
            )
        end
        # ϵₓ = mean(
        #     let
        #         zₛ = sort(z[d,:])
        #         maximum( 
        #             let
        #                 f = (i-1)/(size(z,2)-1)
        #                 if s < -10
        #                     f
        #                 elseif s > 10
        #                     1-f
        #                 else
        #                     abs((s+10)/(2*10) - f)
        #                 end
        #             end for (i,s) in enumerate(zₛ) 
        #         )
        #     end for d ∈ 1:min(2,size(z,1))
        # )

        # ϵₙ = let
        #     dₛ = sort(.√d̂)
        #     qd = Zygote.dropgrad(distance_quantiles(length(dₛ)))
        #     mean( (dₛ .- qd).^2 ) / Lₛ^2
        # end
        
        # ϵₙ = 1 - cov(d̂, d)/(std(d̂)*std(d))

        # ϵₙ = mean(
        #     let
        #         d, d̂ = D̄²[:,j], D̂²[:,j]
        #         f    = 20/length(d)
        #         n, n̂ = softrank(d ./ mean(d)) .≤ f, softrank(d̂ ./ mean(d̂)) .≤ f
        #         sum(n) + sum(n̂) - 2*sum(n.*n̂)
        #     end for j ∈ 1:size(D̂²,2)
        # )
        
        # distance softranks
        # ϵₙ = let
        #     r, r̂ = softrank(d ./ mean(d)), softrank(d̂ ./ mean(d̂))
        #     1 - cov(r, r̂)/(std(r)*std(r̂))
        # end
        ϵₛ = mean(
            let
                d, d̂ = D̄²[:,j], D̂²[:,j]
                r, r̂ = softrank(d ./ mean(d)), softrank(d̂ ./ mean(d̂))
                1 - cov(r, r̂)/(std(r)*std(r̂))
            end for j ∈ 1:size(D̂²,2)
        )

        # ϵₛ = std([sum( (δ[1][1] .≤ z[1,:] .≤ δ[1][2]) .& (δ[2][1] .≤ z[2,:] .≤ δ[2][2])) for δ in Δ]) / size(z,2)
        # ϵₛ = let
        #     x = Point2.(z[1,:], z[2,:])
        #     std(voronoiarea(voronoicells(x,rect)))
        # end
        
        if log
            @show ϵᵣ, ϵₛ, ϵₓ
        end

        return ϵᵣ + param.γₛ*ϵₛ + param.γₓ*ϵₓ #+ param.γₙ*ϵₙ
     end
end

# ------------------------------------------------------------------------
# main functions

function run(data, genes, param; D²=nothing, F=nothing, dₒ=35)
    x⃗, ω, ϕ = preprocess(data; dₒ=dₒ, F=F)

    D² = isnothing(D²) ? geodesics(ϕ(x⃗), param.kₙ).^2 : D²
    kₙ = param.kₙ

    M = model(size(x⃗, 1), param.dₒ; 
          Ws         = param.Ws, 
          normalizes = param.BN, 
          dropouts   = param.DO
    )

    y⃗, I = validate(x⃗, param.V)

    R          = ball(D², param.kₙ)
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

function continuerun(result::Result, input, epochs)
    loss = buildloss(result.model, input.D², result.param)
    train!(result.model, input.y.train, input.index.train, loss; 
        η   = result.param.η, 
        B   = result.param.B, 
        N   = epochs,
        log = input.log
    )

    return result, input
end

# TODO: use above function to simplify
function run(params, niter::Int)
    println("loading data...")

    # scrna, genes = embed(pointcloud(;α=1/500, δ=15), 50; σ=.02), nothing
    scrna, genes = expression()
    x⃗, ω, ϕ      = preprocess(scrna; dₒ=35) #, ϕ=sqrt)

    println("computing geodesics...")

    D², kₙ  = geodesics(ϕ(x⃗), params[1].kₙ).^2, params[1].kₙ
    results = Array{Result}(undef, length(params)*niter)

    for (iₚ, p) in enumerate(params)
        for iᵢₜ in 1:niter
            M = model(size(x⃗, 1), p.dₒ; Ws = p.Ws, normalizes = p.BN, dropouts = p.DO)
            y⃗, I = validate(x⃗, p.V)

            if p.kₙ != kₙ
                D², kₙ = geodesics(ϕ(x⃗), p.kₙ).^2, p.kₙ
            end

            R = ball(D², p.kₙ)
            loss = (x, i, log) -> begin
                z = M.pullback(x)
                x̂ = M.pushforward(z)

                # reconstruction
                ϵᵣ = sum(sum((x.-x̂).^2, dims=2).*ω)/size(x,2)

                # pairwise distances
                D̂² = distance²(z)
                D̄² = D²[i,i]
                # R = ball(D̄², p.kₗ)
                
                d = upper_tri(D̄²)
                d̂ = upper_tri(D̂²)

                # neighborhood isometry
                # n  = (d .<= R) .| (d̂ .<= R) # n  = n .| (d̂ .<= R)
                # nₑ = sum(n)
                # ϵₓ = sum( (d[n] .- d̂[n]).^2 ) / (nₑ * R^2)
                # ϵₓ = sum( (d[n] .- d̂[n]).^2 ) / (sum(d[n])*sum(d̂[n]))

                ϵₓ = sum( (d .- d̂).^2 ) / (length(d) * R^2)

                # distance softranks
                # r, r̂ = softrank(d ./ mean(d)), softrank(d̂ ./ mean(d̂))
                # ϵₛ = 1 - cov(r, r̂)/(std(r)*std(r̂))
                
                if log
                    # @show sum(n)/length(n)
                    @show ϵᵣ, ϵₓ
                end

                return ϵᵣ + p.γₓ*ϵₓ + p.γₛ*ϵₛ
            end

            E = (
                train = zeros(p.N÷p.δ),
                valid = zeros(p.N÷p.δ),
            )
            log  = (n) -> begin
                if (n-1) % p.δ == 0 
                    @show n

                    E.train[(n-1)÷p.δ+1] = loss(y⃗.train, I.train, (n-1) % 10 == 0)
                    E.valid[(n-1)÷p.δ+1] = loss(y⃗.valid, I.valid, (n-1) % 10 == 0)
                end

                nothing
            end

            train!(M, y⃗.train, loss; η=p.η, B = p.B, N = p.N, log = log)

            results[niter*(iₚ-1)+iᵢₜ] = Result(p, E, M)
        end
    end

    return results, (x=x⃗, raw=scrna, genes=genes, map=ϕ) 
end

# run(param...) = run(collect(param), 1)

function main(input, niter, output)
    params = Result[]
    open(input, "r") do io 
        params = eval(Meta.parse(read(io, String)))
    end
    
    result, data = run(params, niter)
    @save "$root/result/$output.bson" result data
end

end
