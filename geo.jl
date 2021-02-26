module PointCloud

import Base:
    eltype, length, minimum, take!

using LinearAlgebra

import ChainRulesCore: rrule, NO_FIELDS

include("queue.jl")
using .PriorityQueue

export distance², distance, embed, upper_tri
export neighborhood, geodesics, mds, isomap

# ------------------------------------------------------------------------
# globals

const ∞ = Inf

# ------------------------------------------------------------------------
# utility functions

distance²(x) = sum( (x[i,:]' .- x[i,:]).^2 for i ∈ 1:size(x,1) )
distance(x)  = .√(distance²(x))

function distance²!(D², x)
    for i ∈ 1:size(D²,2)
        for j ∈ 1:(i-1)
            @inbounds D[i,j] = D[j,i] = sum((x[:,i] - x[:,j]).^2)
        end
    end
end

upper_tri(x) = [ x[i.I[1], i.I[2]] for i ∈ CartesianIndices(x) if i.I[1] < i.I[2] ]
function rrule(::typeof(upper_tri), m)
	x = upper_tri(m)
	return x, (∇) -> begin
		∇m = zeros(size(m))
		n = 1
		for i ∈ CartesianIndices(∇m)
			if i.I[1] >= i.I[2]
				continue
			end
			
			∇m[i.I[1],i.I[2]] = ∇[n]
			∇m[i.I[2],i.I[1]] = ∇[n]

			n += 1
		end
		
        (NO_FIELDS, ∇m)
    end
end

function embed(x, dₒ; σ=0.00)
	y = if dₒ > size(x,1)
			vcat(x, zeros(dₒ-size(x,1), size(x,2)))
		elseif dₒ == size(x,1)
		    x	
		else
			error("cannot embed into smaller dimension")
		end
	
	return y .+ σ.*randn(size(y)...)
end

# ------------------------------------------------------------------------
# point cloud generation functions

function sphere(N; R=1)
	θ = π  .* rand(Float64, N)
	ϕ = 2π .* rand(Float64, N)
	
    return R.*hcat(sin.(θ).*cos.(ϕ), sin.(θ).*sin.(ϕ), cos.(θ))', hcat(θ, ϕ)'
end

function spherical_distance(x; R=1)
    n̂ = x / R
    D = zeros(size(x,2),size(x,2))

    norm² = sum(n̂[i,:].^2 for i ∈ 1:size(x,1))
    cosΔψ = sum(n̂[i,:]' .* n̂[i,:] for i ∈ 1:size(x,1))
    sinΔψ = [norm(cross(n̂[:,i], n̂[:,i])) for i ∈ 1:size(x,2), j ∈ 1:size(x,2)]
    
    return R*atan.(sinΔψ, cosΔψ)
end

function swiss_roll(N; z₀=10, R=1/20)
    z = (z₀/R)*rand(Float64, N)
	ϕ = 1.5π .+ 3π .* rand(Float64, N)
	
    return hcat(ϕ .* cos.(ϕ), ϕ .* sin.(ϕ), z)' .* R, hcat(ϕ, z)'
end

# XXX: calculate geodesics?

function torus(N; R=2, r=1)
	θ = 2π  .* rand(Float64, N)
	ϕ = 2π .* rand(Float64, N)
	
	return hcat((R .+ r*cos.(θ)) .* cos.(ϕ), (R .+ r*cos.(θ)).* sin.(ϕ), r*sin.(θ))', hcat(ϕ, θ)'
end

# XXX: calculate geodesics?

# ------------------------------------------------------------------------
# types for neighborhood graph

struct Vertex{T <: Real}
    position :: Array{T}
end
Vertex(x) = Vertex{eltype(x)}(x)

eltype(v::Vertex{T}) where T <: Real = T

struct Edge{T <: Real}
    verts    :: Tuple{Int, Int}
    distance :: T
end
Edge(verts, distance) = Edge{typeof(distance)}(verts, distance)

eltype(e::Edge{T}) where T <: Real = T

struct Graph{T <: Real}
    verts :: Array{Vertex{T}, 1}
    edges :: Array{Edge{T}, 1}
end
Graph(verts::Array{Vertex{T},1}) where T <: Real = Graph{T}(verts, [])

length(G::Graph) = length(G.verts)

# ------------------------------------------------------------------------
# operations

function neighborhood(x, k::Int)
    D = distance(x)

    G = Graph([Vertex(x[:,i]) for i ∈ 1:size(x,2)])
    for i ∈ 1:size(D,1)
        neighbor = sortperm(D[i,:])[2:end]
        append!(G.edges, [Edge((i,j), D[i,j]) for j ∈ neighbor[1:k]])
    end

    return G
end

function adjacency_list(G::Graph)
    adj = [ Tuple{Int, Float64}[] for v ∈ 1:length(G.verts) ]
    for e ∈ G.edges
        v₁, v₂ = e.verts
        push!(adj[v₁], (v₂, e.distance))
        push!(adj[v₂], (v₁, e.distance))
    end

    return adj
end

function dijkstra!(dist, adj, src)
    dist      .= ∞
    dist[src]  = 0

    Q = RankedQueue((src, 0.0))
    sizehint!(Q, length(dist))

    while length(Q) > 0
        u, d₀ = take!(Q)
        for (v, d₂) ∈ adj[u]
            d₀₂ = d₀ + d₂
            if d₀₂ < dist[v]
                dist[v] = d₀₂
                if v ∈ Q
                    update!(Q, v, d₀₂)
                else
                    insert!(Q, v, d₀₂)
                end
            end
        end
    end
end

function floyd_warshall(G::Graph)
    V = length(G.verts)
    D = fill(∞, (V,V))
    # remove diagonal
    for ij ∈ CartesianIndices(D)
        if ij.I[1] == ij.I[2]
            D[ij] = 0
        end
    end

    # all length 1 paths
    for e ∈ G.edges
        D[e.verts[1],e.verts[2]] = e.distance
        D[e.verts[2],e.verts[1]] = e.distance
    end

    # naive V³ paths
    for k ∈ 1:V
        for i ∈ 1:V
            for j ∈ 1:V
                if D[i,j] > D[i,k] + D[k,j]
                    D[i,j] = D[i,k] + D[k,j]
                    D[j,i] = D[i,j]
                end
            end
        end
    end

    return D
end

function geodesics(G::Graph; sparse=true)
    if sparse
        adj  = adjacency_list(G)
        dist = zeros(length(G), length(G))
        # uncomment for parallelism
        # Threads.@threads
        for v ∈ 1:length(G)
            dijkstra!(view(dist,:,v), adj, v)
        end

        return dist
    else
        return floyd_warshall(G)
    end
end
geodesics(x, k; sparse=true) = geodesics(neighborhood(x, k); sparse=sparse)

# ------------------------------------------------------------------------
# non ml dimensional reduction

function mds(D², dₒ)
    N = size(D²,1)
    C = I - fill(1/N, (N,N))
    B = -1/2 * C*D²*D

    eig = eigen(B)
    ι   = sortperm(λ.values; rev=true)

    λ = eig.values[ι]
    ν = eig.vectors[:,ι]

    return ν[:,1:dₒ] * Diagonal(sqrt.(λ[1:dₒ]))
end

function isomap(x, dₒ; k=12)
    G = neighborhood(x, k)
    D = geodesics(G)
    return mds(D.^2, dₒ)
end

# ------------------------------------------------------------------------
# tests

using Statistics

function test()
    r, ξ = sphere(2000)
    D = spherical_distance(r)

    r = embed(r, size(r,2); σ=0.1)
    ks = collect(4:2:50) 
    ρ  = zeros(length(ks))

    for (i,k) ∈ enumerate(ks)
        D̂ = geodesics(r, k)
        ρ[i] = cor(upper_tri(D̂), upper_tri(D))
    end

    return ks, ρ
end

end
