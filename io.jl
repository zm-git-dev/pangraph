module DataIO

using Match
using GZip

export read_ply, read_matrix

# ------------------------------------------------------------------------
# globals

const root = "/home/nolln/root/data/seqspace"

panic(msg) = error(msg)

# ------------------------------------------------------------------------
# geometric file formats

plytype = Dict{String, Type}(
    "char"   => Int8,    "int8"    => Int8,
    "short"  => Int16,   "int16"   => Int16,
    "int"    => Int32,   "int32"   => Int32,
    "long"   => Int64,   "int64"   => Int64,

    "uchar"   => UInt8,  "uint8"   => UInt8,
    "ushort"  => UInt16, "uint16"  => UInt16,
    "uint"    => UInt32, "uint32"  => UInt32,
    "ulong"   => UInt64, "uint64"  => UInt64,

    "float"  => Float32, "float32" => Float32,
    "double" => Float64, "float64" => Float64
)

readwords(io::IO) = split(readline(io))

struct PLYProp
    name :: Symbol
    type :: Type
    len  :: Union{Type,Nothing}
end

PLYProp(word) = if word[1] == "list"
                    PLYProp(Symbol(word[4]), plytype[word[3]], plytype[word[2]])
                else
                    PLYProp(Symbol(word[2]), plytype[word[1]], nothing)
                end

function fill!(prop::Array{PLYProp}, io::IO)
    word = readwords(io)
    while word[1] == "property" && length(word) >= 3
        push!(prop, PLYProp(word[2:end]))
        word = readwords(io)
    end

    return word
end

function readprops(io::IO, props::Array{PLYProp,1})
    data = (p) -> begin
        if isnothing(p.len)
            return read(io,p.type)
        else
            len = read(io,p.len)
            return [read(io,p.type) for _ in 1:len]
        end
    end

    return (; (prop.name => data(prop) for prop in props)...)
end

function read_ply(io::IO)
    # magic
    line = readline(io)
    if line != "ply"
        panic("can not interpret byte stream as PLY data")
    end

    # file format
    word = readwords(io)
    if word[1] != "format" || word[3] != "1.0"
        panic("unrecognized format of PLY file")
    end
    # FIXME: we just ignore this for now and assume little endian binary
    fmt = Symbol(word[2])

    # data layout
    n₀, n₂ = 0, 0
    field₀ = PLYProp[] # list of fields of vertices
    field₂ = PLYProp[] # list of fields of faces

    @label getline # ----------------------------
    word = readwords(io)

    @label doline
    if word[1] == "end_header" 
        @goto endloop
    end

    if word[1] != "element" || length(word) != 3
        @goto getline
    end

    @match word[2] begin
        "vertex" => begin
            n₀   = parse(Int, word[3])
            word = fill!(field₀, io)
            @goto doline
        end
        "face" => begin
            n₂   = parse(Int, word[3])
            word = fill!(field₂, io)
            @goto doline
        end
    end

    @goto  getline

    @label endloop # ----------------------------

    return (
        verts = [readprops(io,field₀) for _ in 1:n₀], 
        faces = [readprops(io,field₂) for _ in 1:n₂]
    )
end

# ------------------------------------------------------------------------
# scRNAseq file formats

function read_matrix(io::IO; type=Float64, named_cols=false, named_rows=false)
    cols  = named_cols ?  readwords(io) : nothing
    ncols = length(cols)

    x = position(io)
    nrows = sum(1 for row in eachline(io))
    seek(io, x)

    data = Array{type, 2}(undef, nrows, ncols)
    rows = named_rows ? Array{String, 1}(undef, nrows) : nothing

    for (i,row) in enumerate(eachline(io))
        entry = split(row)
        if named_rows
            rows[i] = entry[1]
            entry   = entry[2:end]
        end
        data[i,:] = parse.(type, entry) 
    end

    return data, rows, cols
end

# ------------------------------------------------------------------------
# point of entry for testing

function test()
    mesh = open("$root/gut/mesh_apical_stab_000153.ply") do io
        read_ply(io)
    end

    bdntp, genes = GZip.open("$root/dvex/bdtnp.txt.gz") do io
        data, _, cols = read_matrix(io; named_cols=true)
        return data, cols
    end

    scrna, genes = GZip.open("$root/dvex/dge_normalized.txt.gz") do io
        data, rows, _ = read_matrix(io; named_rows=true, named_cols=true)
        return data, rows
    end
end

end
