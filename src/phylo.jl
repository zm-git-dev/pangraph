module Phylo

using Rematch

# ------------------------------------------------------------------------
# types

struct Tree
    name   :: String
    dist   :: Float64
    parent :: Union{Tree,Nothing}
    child  :: Array{Tree,1}

    data
end

# input/output

# ------------------------------------------------------------------------
# methods

function binary!(root::Tree)
    return root
end

function reroot!(root::Tree, outgroup::Tree)
    return root
end

function rescale!(root::Tree, by::T) where T <: Real
    for node in root.traverse("preorder")
        node.dist *= by
    end
end

function write(io::IO, root::Tree)
    Base.write(io, root.write(format=1))
end

function dictionary(root::Tree, level::Int, i::Int; mutations=false)
    node = Dict(
        "name" => if root.name == ""
            i += 1
            suffix = string(i-1)
            suffix = @match length(suffix) begin
                1 => "0000$(suffix)"
                2 => "000$(suffix)"
                3 => "00$(suffix)"
                4 => "0$(suffix)"
                _ => suffix
            end
            "NODE_$(suffix)"
        else
            root.name
        end,
        "branch_length" => root.dist,
    )

    if mutations
        # TODO: actually write mutations?
        node["muts"]    = ""
        node["aa_muts"] = ""
    else
        node["clade"] = level
    end

    if root.name != ""
        if mutations
            node["accession"]  = split(root.name,'#')[1]
            node["annotation"] = "pan-contig"
        else
            node["attr"] = Dict(
                "host"   => root.name,
                "strain" => root.name,
            )
        end
        return node, i
    end

    node["children"] = Dict[]
    for child in root.children
        c, i = dictionary(child, level+1, i; mutations=mutations)
        push!(node["children"], c)
    end

    return node, i
end

end
