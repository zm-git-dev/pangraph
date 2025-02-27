module Commands

using ..PanGraph: panic

export Arg, Command
export flags, usage, arg

Maybe{T} = Union{Missing,T}

mutable struct Arg
    type::Type
    meta::AbstractString
    flag::NamedTuple{(:short,:long), Tuple{Maybe{AbstractString},Maybe{AbstractString}}}
    help::AbstractString

    value
end

flags(arg::Arg) =
    if !ismissing(arg.flag.long) && !ismissing(arg.flag.short) 
        "$(arg.flag.long), $(arg.flag.short)"
    elseif !ismissing(arg.flag.long) && ismissing(arg.flag.short) 
        "$(arg.flag.long)"
    elseif ismissing(arg.flag.long) && !ismissing(arg.flag.short) 
        "$(arg.flag.short)"
    else 
        ""
    end

mutable struct Command
    cmd::AbstractString
    use::AbstractString
    meta::AbstractString
    foot::AbstractString

    sub::Array{Command,1}
    arg::Array{Arg,1}

    run::Maybe{Function}
end

Command(cmd,help,meta,foot,arg::Array{Arg})     = Command(cmd,help,meta,foot,Command[],arg,missing)
Command(cmd,help,meta,foot,sub::Array{Command}) = Command(cmd,help,meta,foot,sub,Arg[],missing)

Command(cmd,help,meta,foot,arg::Array{Arg},fn::Function)     = Command(cmd,help,meta,foot,Command[],arg,fn)
Command(cmd,help,meta,foot,sub::Array{Command},fn::Function) = Command(cmd,help,meta,foot,sub,Arg[],fn)

function usage(io::IO, cmd::Command)
    print(io,"""
    usage: $(cmd.use)

        $(cmd.meta)

    """)

    if length(cmd.sub) > 0
        println(io,"The available commands are:\n")

        maxlen = maximum(length(sub.cmd) for sub in cmd.sub)
        for sub in cmd.sub
            println(io, "\t$(sub.cmd)$(" "^(maxlen-length(sub.cmd)))\t$(sub.meta)")
        end

		println(io,"")
    end

    if length(cmd.arg) > 0
        println(io,"The available options are:\n")

        maxlen = maximum(length(flags(arg)) for arg in cmd.arg)
        for (i,arg) in enumerate(cmd.arg)
            flag = flags(arg)
            println(io, "\t$(arg.help)")
            println(io, "\t$(flag)$(" "^(maxlen-length(flag)))\t$(arg.meta)")
            println(io, "\t$(" "^maxlen)\t$(arg.type) default: $(arg.value)")
            if i < length(cmd.arg)
                println("")
            end
        end

		println(io,"")
    end

    if length(cmd.foot) > 0
        println(io,"The expected arguments are:\n")
        println(io,"\t", join(split(cmd.foot,"\n"), "\n\t"))
    end

	return 2
end

usage(cmd::Command) = usage(stderr, cmd)

function Base.parse(cmd::Command, args)
    for sub in cmd.sub
        if sub.cmd == args[1]
            cmd.run = sub.run
            return args[2:end]
        end
    end

    if length(cmd.sub) > 0
        if length(cmd.sub) ≥ 1 && args[1] == "-h"
            usage(cmd)
            exit(2)
        end
        panic("""
        pangraph $(args[1]): unknown command
        Run 'pangraph help' for usage.
        """)
    end

	length(cmd.arg) > 0 || return length(args) > 0 ? args : nothing

    let arg = ""
        itr = Iterators.Stateful(args)
        if isempty(itr)
            usage(cmd)
            return nothing
        end

        arguments = String[]
        @label ARGLOOP #-------------------------
        !isempty(itr) || @goto ENDLOOP

        word = peek(itr)
        if !startswith(word, "-")
            push!(arguments,popfirst!(itr))
            @goto ARGLOOP
        end

        popfirst!(itr) # advance iterator
        for arg ∈ cmd.arg
            if arg.flag.long == word || arg.flag.short == word
                arg.value =
                if arg.type == Bool
                    true
                else
                    if isempty(itr)
                        panic("""flag '$(word)' requires non-missing input data of type '$(arg.type)'
                        Run 'pangraph help' for usage.
                        """)
                    end

                    val = popfirst!(itr)
                    try
                        arg.type == String ? val : Base.parse(arg.type, val)
                    catch e
                        if isa(e, ArgumentError)
                            panic("""flag '$(word)' requires input data of type '$(arg.type)'. recieved '$(val)'
                            Run 'pangraph help' for usage.
                            """)
                        else
                            throw(e)
                        end
                    end
                end

                @goto ARGLOOP
            end
        end

        if word == "-h"
            usage(cmd)
            exit(2)
        end

        panic("""provided flag '$(word)' not expected
        Run 'pangraph help' for usage.
        """)

        @label ENDLOOP #-------------------------

        return arguments
    end
end

function Base.run(cmd::Command, args) 
    if ismissing(cmd.run)
        error("command $(cmd.cmd) does not have a valid execution function")
    end
    cmd.run(args)
end

function arg(cmd::Command, short::AbstractString)
    i = 1
    while i ≤ length(cmd.arg) && cmd.arg[i].flag.short != short
        i += 1
    end
    cmd.arg[i].flag.short != short && error("flag $(short) not found")

    return cmd.arg[i].value
end

end
