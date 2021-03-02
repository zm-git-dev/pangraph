using Flux
using BSON: @load

include("src/main.jl")
using .SeqSpace

function load(name)
    @load "$(root)/result/$name.bson" result
    return result
end

function names()
    return map(readdir("$(root)/result")) do fd
        join(split(basename(fd), ".")[1:end-1], ".")
    end
end

function partition(results; field=:Ws)
    parts = Dict{typeof(getproperty(results[1].param, field)), Array{Result,1}}()
    for r in results
        key = getproperty(r.param, field)
        if key ∉ keys(parts)
            parts[key] = [r]
        else
            push!(parts[key], r)
        end
    end

    return parts
end

module Plot
    using Plots
    using ColorSchemes

    new() = plot()

    function grid(results)
        # TODO: fill in when grid results finish
        @show map(results) do r
            r.param.Ws
        end
    end

    function losses(results; kwargs...)
        p = new()
        losses!(results; kwargs...)
        p
    end

    function losses!(results; cmap=ColorSchemes.Paired_10)
        for (i,r) in enumerate(results)
            x = 1:r.param.δ:r.param.N
            plot!(x, r.loss.train,
                linecolor = get(cmap, i/length(results)),
                linestyle = :solid,
                legend    = false,
            )
            plot!(x, r.loss.valid,
                linecolor = get(cmap, i/length(results)),
                linestyle = :dashdot,
                legend    = false,
            )
        end
    end
end

# ------------------------------------------------------------------------
# main point of entry

if abspath(PROGRAM_FILE) == @__FILE__
    @show ARGS
    Analyze.load("grid")
end
