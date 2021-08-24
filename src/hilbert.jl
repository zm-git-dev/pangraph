module Hilbert

using HilbertSpaceFillingCurve

import ChainRulesCore: rrule, NO_FIELDS

# assumes points between -1 and +1
absrank(x::Array{T,2}) where T <: Real = [
    let
        p = round.(Int, typemax(UInt16).*(col .+ 1)./2)
        hilbert(p, 2)
    end for col in eachcol(x) 
]

positions(rank) = hcat(
    ( 2*(hilbert(round(Int,typemax(UInt32)*r), 2)/typemax(UInt16)) .- 1 for r in rank )...
)

function sort(x)
    r = absrank(x)
    ι = sortperm(r)
    return x[:,ι]
end

function rrule(::typeof(sort), x)
    r = absrank(x)
    ι = sortperm(r)
    return x[:,ι], (∇) -> (NO_FIELDS, ∇[:,ι])
end

end
