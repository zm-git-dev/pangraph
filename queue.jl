# Used for Dijisktra's algorithm
module PriorityQueue

import Base:
    ∈, length, minimum, 
    push!, take!, insert!

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


