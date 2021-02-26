module SeqSpace

include("geo.jl")
include("io.jl")
include("model.jl")

using .PointCloud, .DataIO, .ML

const root = "/home/nolln/root/data/seqspace"

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

mean(x)    = sum(x) / length(x)
ball(D, k) = mean(sort(view(D,:,i))[k+1] for i in 1:size(D,2))

const N  = 500
const k  = 5
const dₒ = 2
const Ws = [50,50]
const γ₀ = .01

struct HyperParameters
    d₀ :: Int          # Output dimensionality
    Ws :: Array{Int,1} # Network layer widths
    N  :: Int          # Number of epochs to run
    V  :: Int          # Number of points to partition for validation
    kₙ :: Int          # Number of neighbors to use to estimate geodesics
    kₗ :: Int          # Average number of neighbors to impose isometry on 
    γ  :: Float64      # Prefactor of neighborhood isometry loss
end

function main()
    ptcloud = embed(pointcloud(; α=1/500, δ=10), 50; σ = .05)
    x⃗, ω, ϕ = preprocess(ptcloud)

    M = model(size(x⃗,1), dₒ; Ws = Ws)
    y⃗, I = validate(x⃗, 128)

    D² = geodesics(ϕ(x⃗), 12).^2
    loss = (x, i) -> begin
        z = M.pullback(x)
        x̂ = M.pushforward(z)

        # reconstruction
        ϵᵣ = sum(sum((x.-x̂).^2, dims=2).*ω)/size(x,2)

        # neighborhood isometry
        D̂² = distance²(z) # XXX: too much allocation in inner loop?

        R = ball(D²[i,i], k)

        d = upper_tri(D²[i,i])
        d̂ = upper_tri(D̂²)

        n = d .<= R
        n̂ = d̂ .<= R

        ϵₓ = .5*(sqrt(mean( (d[n] .- d̂[n]).^2 )) 
               + sqrt(mean( (d̂[n̂] .- d[n̂]).^2 )))

        return ϵᵣ + γ₀*ϵₓ
    end

    E = (
        train = zeros(N÷5),
        valid = zeros(N÷5),
    )
    log  = (n, loss, model) -> begin
        if (n-1) % 5 == 0 
            E.train[(n-1)÷5+1] = loss(y⃗.train, I.train)
            E.valid[(n-1)÷5+1] = loss(y⃗.valid, I.valid)
        end

        nothing
    end

    train!(M, y⃗.train, loss; η=1e-3, N = N, log = log)

    return (
        data  = x⃗,
        model = M,
        loss  = E,
    )
end

end
