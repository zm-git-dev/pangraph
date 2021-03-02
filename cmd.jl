include("src/main.jl")

function cmd()
    args = ARGS
    length(args) != 3 && error("invalid number of input parameters")

    SeqSpace.main(args[3], parse(Int, args[2]), args[1])
end

# ------------------------------------------------------------------------
# main point of entry

if abspath(PROGRAM_FILE) == @__FILE__
    cmd()
end
