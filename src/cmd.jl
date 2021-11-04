module Shell

using ..Graphs: marshal, serialize

export minimap2, mash, mafft

# command line execution
function execute(cmd::Cmd; now=true)
    out = Pipe()
    err = Pipe()

    proc = run(pipeline(cmd, stdout=out, stderr=err); wait=now)

    close(out.in)
    close(err.in)

    stdout = @async String(read(out))
    stderr = @async String(read(err))

    if now
        return (
            out  = fetch(stdout),
            err  = fetch(stderr),
            code = proc.exitcode, #err,
        )
    else
        return (
            out  = stdout,
            err  = stderr,
            proc = proc,
        )
    end
end

function mash(Gs...)
    function compare(path)
        result = execute(`mash triangle $path`; now=false)
        stdout = IOBuffer(fetch(result.out))

        N     = parse(Int64,readline(stdout))
        dist  = zeros(N,N)
        names = Array{String}(undef, N)
        for (i, line) in enumerate(eachline(stdout))
            elt = split(strip(line))
            names[i] = elt[1]
            dist[i,1:(i-1)] = [parse(Float64,x) for x in elt[2:end]]
        end

        dist = dist + dist';

        return dist, names
    end

    distance, names = mktemp() do path, io
        writer = @async let
            for (i,G) ∈ enumerate(Gs)
                serialize(io, G)
            end
        end

        reader = @async compare(path)

        wait(writer)
        distance, names = fetch(reader)
    end
end

function minimap2(qry::String, ref::String)
    return execute(`minimap2 -x asm10 -m 10 -n 1 -s 30 -D -c $ref $qry`; now=false)
end

function mafft(block)
    out = IOBuffer()
    aln = IOBuffer()

    names = marshal(aln, block, :fasta)
    aln = IOBuffer(String(take!(aln)))

    run(pipeline(pipeline(
        aln, `mafft --auto --nuc /dev/stdin`, out
        ); stderr=devnull ); wait=true
    )

    close(aln)

    return IOBuffer(String(take!(out))), names
end

end
