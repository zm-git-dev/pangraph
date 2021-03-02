module SeqSpace

using GZip
using BSON: @save

include("geo.jl")
include("io.jl")
include("model.jl")

using .PointCloud, .DataIO, .ML

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
    γ  :: Float64      # prefactor of neighborhood isometry loss
end

HyperParams(; dₒ=2, Ws=Int[], BN=Int[], DO=Int[], N=1000, δ=5, η=1e-3, B=64, V=64, kₙ=12, kₗ=3, γ=.01) = HyperParams(dₒ, Ws, BN, DO, N, δ, η, B, V, kₙ, kₗ, γ)

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

function expression()
    scrna, genes, _ = GZip.open("$root/dvex/dge_normalized.txt.gz") do io
        read_matrix(io; named_cols=true, named_rows=true)
    end

    return scrna, genes
end

mean(x)    = sum(x) / length(x)
median(x)  = sort(x)[length(x)÷2+1]
ball(D, k) = mean([sort(view(D,:,i))[k+1] for i in 1:size(D,2)])

# ------------------------------------------------------------------------
# main functions

function run(params, niter::Int)
    println("loading data...")

    scrna, genes = expression()
    x⃗, ω, ϕ      = preprocess(scrna; dₒ=50, ϕ=sqrt)

    @show size(x⃗), size(ϕ(x⃗))

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

            loss = (x, i, log) -> begin
                z = M.pullback(x)
                x̂ = M.pushforward(z)

                # reconstruction
                ϵᵣ = sum(sum((x.-x̂).^2, dims=2).*ω)/size(x,2)

                # neighborhood isometry
                # D̂² = distance²(z)
                # D̄² = D²[i,i]

                # R = ball(D̄², p.kₗ)
                # d = upper_tri(D̄²)
                # d̂ = upper_tri(D̂²)

                # n = (d .<= R) .| (d̂ .<= R)

                # ϵₓ = mean( (d[n] .- d̂[n]).^2 ) / R^2
                if log
                    #@show sum(n)/length(n)
                    @show ϵᵣ #, ϵₓ
                end

                return ϵᵣ #+ p.γ*ϵₓ
            end

            E = (
                train = zeros(p.N÷p.δ),
                valid = zeros(p.N÷p.δ),
            )
            log  = (n, loss, model) -> begin
                if (n-1) % p.δ == 0 
                    @show n

                    E.train[(n-1)÷p.δ+1] = loss(y⃗.train, I.train, true)
                    E.valid[(n-1)÷p.δ+1] = loss(y⃗.valid, I.valid, true)
                end

                nothing
            end

            println("training model...")
            train!(M, y⃗.train, loss; η=p.η, B = p.B, N = p.N, log = log)

            results[niter*(iₚ-1)+iᵢₜ] = Result(p, E, M)
        end
    end

    return results, (x=x⃗, genes=genes, map=ϕ) 
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
