module Phylo

using Rematch

# ------------------------------------------------------------------------
# types

mutable struct Tree
    name     :: String
    comment  :: String
    distance :: Float64
    support  :: Float64
    parent   :: Union{Tree,Nothing}
    child    :: Array{Tree,1}
end
Tree() = Tree("", "", 0, nothing, Tree[])
Tree(parent::Tree) = Tree("", "", 0, parent, Tree[])

# input/output

struct Token
    type :: Char
    data
end
Token(type) = Token(type, nothing)

const lexicon = Set{Char}(['(', ')', '[', ']', ',', ';', ':'])

function isidentchar(c::Char)
    return (
        c == '!' || c == '\\' ||
        ('\"' < c < '\'')     ||
        (')' < c < '+')       ||
        (',' < c < ':')       ||
        (':' < c < '[')       ||
        (']' < c ≤ '~')
    )
end

function lex(io::IO)::Token
    buffer = IOBuffer()
    letter = peek(io, Char)

    # space
    if letter |> isspace
        while letter |> isspace
            write(buffer, read(io, Char))
            letter = peek(io, Char)
        end

        return Token(' ', String(take!(buffer)))
    end

    # simple tokens
    letter ∉ lexicon || return Token(letter)

    # quoted identifier
    if letter == '"'
        read(io, Char) # eat quote
        letter = read(io, Char)

        while letter != '"'
            write(buffer, letter)
            letter = read(io, Char)
        end

        return Token('$', String(take!(buffer)))
    end

    # number
    if letter == '.' || isdigit(letter)
    @label NUMBER
        write(buffer, read(io, Char))
        letter = peek(io, Char)

        while letter |> isdigit
            write(buffer, read(io, Char))
            letter = peek(io, Char)
        end

        letter != '.' || @goto NUMBER
        letter |> isidentchar && @goto IDENTIFIER

        return Token('#',parse(String(take!(buffer)), Float64))
    end

    # identifier
    @label IDENTIFIER
    while letter |> isidentchar
        write(buffer, read(io, Char))
        letter = peek(io, Char)
    end
    return Token('$', String(take!(buffer)))
end

function lex!(io::IO)::Token
    token = lex(io)
    while token.type == ' '
        token = lex(io)
    end
    return token
end

function read(io::IO, parent::Tree)
    node = parent
    while true
        token = lex!(io)
        @match token.type begin
            '(' => let
                global node = Tree(parent)
                push!(parent.child, node)
                token = read(io, node)
                token.kind == ')' || error("syntax error: expected closing paren, found $(token)")
            end
            ')' => let
                error("syntax error: unexpected closing paren found")
            end
            '[' => let
                error("syntax error: newick comments not supported")
            end
            ']' => let
                error("syntax error: unexpected closing bracket found")
            end
            ':' => let
                token = lex!(io)
                token.kind == '#' || error("syntax error: expected number following ':'")
                node !== nothing  || error("syntax error: setting distance of nil node")

                node.distance = token.data
            end
            ',' => let
                global node = nothing
            end
            ';' => let
            end
            '$' => let
            end
            '#' => let
            end
             _  => error("unrecognized token type: $(token.type)")
        end
    end
end

function read(io::IO)
    root = Tree()
    read(io, root)
    return root
end

function write(io::IO)
end

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
