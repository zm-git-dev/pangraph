module WFMash

import ..PanGraph: PanContigs, Alignment
import ..PanGraph.Graphs.Utility: read_paf, write_fasta, uncigar
import ..PanGraph.Graphs.Shell: execute

using Random

export align

"""
    recigar!(hit::Alignment)

Transform the detailed cigar string returned from wfmash into the more conventional form returned by minimap2.
Wfmash returns detailed match/mismatch information that we do not need.
Merges them into one match category.
"""
function recigar!(hit::Alignment)
    buffer = IOBuffer()
    n = 0
    i₁, i₂ = 1, 1
    while i₁ ≤ length(hit.cigar)
        while i₂ ≤ length(hit.cigar) && isdigit(hit.cigar[i₂])
            i₂ += 1
        end

        if hit.cigar[i₂] == '=' || hit.cigar[i₂] == 'X'
            n += parse(Int, hit.cigar[i₁:i₂-1])
        else
            if n > 0
                print(buffer, "$(n)M")
                n = 0
            end
            print(buffer, hit.cigar[i₁:i₂])
        end

        i₁ = i₂ + 1
        i₂ = i₁
    end

    if n > 0
        print(buffer, "$(n)M")
        n = 0
    end

    hit.cigar = String(take!(buffer))
    hit.cigar = collect(uncigar(hit.cigar))
    hit = trim_ending!(hit)
    return hit
end

"""
    trim_ending!(hit::Alignment)

Remove leading and trailing deletions and insertions from the cigar,
so that the first and last entry is a match.
"""
function trim_ending!(hit::Alignment)
    
    # trim leading deletions/insertions
    while hit.cigar[1][2] != 'M'
        len, type = popfirst!(hit.cigar)
        hit.length -= len
        if type == 'I'
            hit.qry.start += len
            if hit.qry.seq !== nothing
                hit.qry.seq = hit.qry.seq[len+1:end]
            end
        elseif type == 'D'
            hit.ref.start += len
            if hit.ref.seq !== nothing
                hit.ref.seq = hit.ref.seq[len+1:end]
            end
        else
            raise("unrecognized cigar type")
        end
    end

    # trim trailing deletions/insertions
    while hit.cigar[end][2] != 'M'
        len, type = pop!(hit.cigar)
        hit.length -= len
        if type == 'I'
            hit.qry.stop -= len
            if hit.qry.seq !== nothing
                hit.qry.seq = hit.qry.seq[1:end-len]
            end
        elseif type == 'D'
            hit.ref.stop -= len
            if hit.ref.seq !== nothing
                hit.ref.seq = hit.ref.seq[1:end-len]
            end
        else
            raise("unrecognized cigar type")
        end
    end

    return hit
end

"""
    align(ref::PanContigs, qry::PanContigs)

Align homologous regions of `qry` and `ref`.
Returns the list of intervals between pancontigs.
"""
function align(ref::PanContigs, qry::PanContigs)
    dir = "wfmash_3/" * randstring(10)
    mkpath(dir)
    println(stderr, "wfmash dir: $dir")
    flush(stderr)
    if ref != qry
        open("$dir/qry.fa","w") do io
            for (name, seq) in zip(qry.name, qry.sequence)
                if length(seq) ≥ 95
                    write_fasta(io, name, seq)
                end
            end
        end

        open("$dir/ref.fa","w") do io
            for (name, seq) in zip(ref.name, ref.sequence)
                if length(seq) ≥ 95
                    write_fasta(io, name, seq)
                end
            end
        end

        run(`samtools faidx $dir/ref.fa`)
        run(`samtools faidx $dir/qry.fa`)
        run(pipeline(`wfmash $dir/ref.fa $dir/qry.fa`,
            stdout="$dir/aln.paf",
            stderr="$dir/err.log"
            )
        )
    else
        open("$dir/seq.fa","w") do io
            for (name, seq) in zip(qry.name, qry.sequence)
                if length(seq) ≥ 95
                    write_fasta(io, name, seq)
                end
            end
        end

        run(`samtools faidx $dir/seq.fa`)
        run(pipeline(`wfmash -X $dir/seq.fa $dir/seq.fa`,
            stdout="$dir/aln.paf",
            stderr="$dir/err.log"
            )
            )
    end
    hits = open(read_paf, "$dir/aln.paf")
    hits = map(recigar!, hits)
    return hits
end

end
