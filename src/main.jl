module SeqSpace

using GZip
using BSON: @save

include("geo.jl")
include("io.jl")
include("rank.jl")
include("model.jl")

using .PointCloud, .DataIO, .SoftRank, .ML

export Result, HyperParams
export run

# ------------------------------------------------------------------------
# globals

# ------------------------------------------------------------------------
# types

struct HyperParams
    dₒ :: Int          # output dimensionality
    Ws :: Array{Int,1} # network layer widths
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
end

HyperParams(; dₒ=3, Ws=Int[], BN=Int[], DO=Int[], N=500, δ=5, η=5e-4, B=64, V=81, kₙ=12, kₗ=3, γₓ=1e-4, γₛ=1) = HyperParams(dₒ, Ws, BN, DO, N, δ, η, B, V, kₙ, kₗ, γₓ, γₛ)

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

ball(D, k) = mean(sort(view(D,:,i))[k+1] for i in 1:size(D,2))

# ------------------------------------------------------------------------
# main functions

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

            R = ball(D², p.kₗ)
            loss = (x, i, log) -> begin
                z = M.pullback(x)
                x̂ = M.pushforward(z)

                # reconstruction
                ϵᵣ = sum(sum((x.-x̂).^2, dims=2).*ω)/size(x,2)

                # pairwise distances
                D̂² = distance²(z)
                D̄² = D²[i,i]
                R = ball(D̄², p.kₗ)
                
                d = upper_tri(D̄²)
                d̂ = upper_tri(D̂²)

                # neighborhood isometry
                n  = (d .<= R) .| (d̂ .<= R)
                # nₑ = sum(n)
                # n  = n .| (d̂ .<= R)
                # ϵₓ = sum( (d[n] .- d̂[n]).^2 ) / (nₑ * R^2)
                ϵₓ = sum( (d[n] .- d̂[n]).^2 ) / (sum(d[n])*sum(d̂[n]))

                # distance softranks
                # r, r̂ = softrank(d ./ mean(d)), softrank(d̂ ./ mean(d̂))
                # ϵₛ   = 1 - cov(r, r̂)/(std(r)*std(r̂))
                
                if log
                    # @show sum(n)/length(n)
                    @show ϵᵣ, ϵₓ#, ϵₛ
                end

                return ϵᵣ + p.γₓ*ϵₓ #+ p.γₛ*ϵₛ
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

            println("training model...")
            train!(M, y⃗.train, loss; η=p.η, B = p.B, N = p.N, log = log)

            results[niter*(iₚ-1)+iᵢₜ] = Result(p, E, M)
        end
    end

    return results, (x=x⃗, raw=scrna, genes=genes, map=ϕ) 
end

run(param...) = run(collect(param), 1)

function main(input, niter, output)
    params = Result[]
    open(input, "r") do io 
        params = eval(Meta.parse(read(io, String)))
    end
    
    result, data = run(params, niter)
    @save "$root/result/$output.bson" result data
end

end
