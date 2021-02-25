module Geometry

import Base:
    eltype, length, minimum, take!

# ------------------------------------------------------------------------
# utility functions

distance²(x) = sum((x[:,i]' .- x[:,i]).^2 for i in 1:size(x,2))
distance(x)  = sqrt.(distance²(x))

upper_tri(x) = [ x[i.I[1], i.I[2]] for i in CartesianIndices(x) if i.I[1] < i.I[2] ]

# ------------------------------------------------------------------------
# types

# Used for Dijisktra's algorithm
module PriorityQueue

import Base:
    ∈,
    length, minimum, push!, take!, insert!

export RankedQueue
export update!

parent(i) = i÷2
left(i)   = 2*i
right(i)  = 2*i+1

struct RankedQueue{T <: Real, S <: Any}
    rank :: Array{T, 1}
    data :: Array{S, 1}
end

ϵ(q::RankedQueue{T,S}, x::S) where {T <: Real, S <: Any} = x ∈ q.data

function rotateup!(q::RankedQueue, i)
    i == 1 && return i

    while i > 1 && (p=parent(i); q.rank[i] < q.rank[p])
        q.rank[i], q.rank[p] = q.rank[p], q.rank[i]
        q.data[i], q.data[p] = q.data[p], q.data[i]
        i = p
    end

    return i
end
rotateup!(q::RankedQueue) = rotateup!(q, length(q.rank))

function rotatedown!(q::RankedQueue, i)
    left(i) > length(q) && return i

    child(i) = (right(i) > length(q) || q.rank[left(i)] < q.rank[right(i)]) ? left(i) : right(i)
    while left(i) <= length(q) && (c=child(i); q.rank[i] > q.rank[c])
        q.rank[i], q.rank[c] = q.rank[c], q.rank[i]
        q.data[i], q.data[c] = q.data[c], q.data[i]
        i = c
    end

    return i
end
rotatedown!(q::RankedQueue) = rotatedown!(q, 1)

function RankedQueue(X::Tuple{S, T}...) where {T <: Real, S <: Any}
    q = RankedQueue{T,S}([x[2] for x in X], [x[1] for x in X])
    i = length(X) ÷ 2
    while i >= 1
        rotatedown!(q, i)
        i -= 1
    end

    return q
end

length(q::RankedQueue) = length(q.rank)

# external methods
# XXX: there be dragons - we don't check for duplicated data being passed to us

minimum(q::RankedQueue) = (data=q.data[1], rank=q.rank[1])

function insert!(q::RankedQueue{T}, data::S, rank::T) where {T <: Real, S <: Any}
    push!(q.rank, rank)
    push!(q.data, data)

    rotateup!(q)
end

function take!(q::RankedQueue)
    r = q.rank[1]
    q.rank[1] = q.rank[end]
    q.rank    = q.rank[1:end-1]

    d = q.data[1]
    q.data[1] = q.data[end]
    q.data    = q.data[1:end-1]

    rotatedown!(q)

    return (data=d, rank=r)
end

function update!(q::RankedQueue{T, S}, data::S, new::T) where {T <: Real, S <: Any}
    (i = findfirst(q.data .== data)) == nothing && panic("attempting to update a non-existent data value")

    old = q.rank[i]
    q.rank[i] = new

    return if new > old
               rotatedown!(q, i)
           elseif new < old
               rotateup!(q, i)
           else
               i
           end
end

end

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
Graph(verts::Array{Vertex,1}) = Graph{eltype(v[1])}(verts, [])

length(G::Graph) = length(G.verts)

# ------------------------------------------------------------------------
# operations

using .PriorityQueue

function neighborhood(x, k::Int)
    D = distance(x)

    G = Graph([Vertex(x[:,i]) for i in 1:size(x,1)])
    for i in 1:size(D,1)
        neighbor = sortperm(D[i,:])[2:end]
        append!(G.edges, [Edge((i,j), D[i,j]) for j in neighbor[1:k]])
    end

    return G
end

function adjacency_list(G::Graph)
    adj = [ Tuple{Int, Float64}[] for v in 1:length(G.verts) ]
    for e in G.edges
        v₁, v₂ = e.verts
        push!(adj[v₁], (v₂,e.distance))
        push!(adj[v₂], (v₁,e.distance))
    end

    return adj
end

function dijkstra!(dist, adj, src)
    dist      .= ∞
    dist[src]  = 0

    Q = RankedQueue((src, 0f0))
    while length(Q) > 0
        v, d₀ = take!(Q)
        d₁    = dist[v]
        for (n, d₂) in adj[v]
            d₁₂ = d₁ + d₂ 
            if d₁₂ < d₀ 
                dist[v] = d₁₂
                if v ∈ Q
                    update!(Q, v, d₁₂)
                else
                    insert!(Q, v, d₁₂)
                end
            end
        end
    end
end

function geodesics(G::Graph)
    adj  = adjacency_list(G)
    dist = zeros(length(G), length(G))
    # uncomment for parallelism
    # Threads.@threads
    for v in 1:length(G)
        dijkstra!(view(dist,:,v), adj, v)
    end

    return dist
end

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

function test()
    Q = RankedQueue((0,1), (1,2), (2,10), (3,23), (4,0))
    @show Q
    insert!(Q, 5, -1)
    @show Q

    nothing
end

end
