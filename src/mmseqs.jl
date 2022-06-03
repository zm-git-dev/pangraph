module MMseqs

import ..PanGraph: PanContigs, Alignment
import ..PanGraph.Graphs.Utility: read_paf, write_fasta, uncigar, read_mmseqs2
import ..PanGraph.Graphs.Shell: execute

export align

function log(vbp, msg...)
    tid, lk = vbp
    lock(lk) do
        print(stderr, "task: $tid from mmseqs align --> ")
        println(stderr, msg...)
        flush(stderr)
    end
end

function myrun(cmd, vbp)
    p = run(cmd; wait=false)
    success(p) || Base.pipeline_error(p)
    log(vbp, "process success: $(success(p))")
end

"""
    align(ref::PanContigs, qry::PanContigs)
Align homologous regions of `qry` and `ref`.
Returns the list of intervals between pancontigs.
"""
function align(ref::PanContigs, qry::PanContigs; vbp=nothing)
    
    
    vb = vbp !== nothing

    hits = mktempdir() do dir
        # hits = let dir = mktempdir(; cleanup = false)
        # log("starting to aling ref and qry in $dir")
        
        vb && log(vbp, "aligning in temp dir: $dir")

        qryfa, reffa = "$dir/qry.fa", "$dir/ref.fa"
        if ref == qry
            reffa = qryfa
        end

        vb && log(vbp, "write qry: $qryfa")

        open("$qryfa", "w") do io
            for (name, seq) in zip(qry.name, qry.sequence)
                if length(seq) ≥ 95
                    write_fasta(io, name, seq)
                end
            end
        end

        if ref != qry

            vb && log(vbp, "create ref: $reffa")

            open("$reffa", "w") do io
                for (name, seq) in zip(ref.name, ref.sequence)
                    if length(seq) ≥ 95
                        write_fasta(io, name, seq)
                    end
                end
            end
        end

        vb && log(vbp, "run easy-search")

        # log("mmseqs search")
        myrun(`mmseqs easy-search
                $qryfa $reffa $dir/res.paf $dir/tmp
                --threads 1 
                -a 
                --max-seq-len 10000
                --search-type 3
                --format-output query,qlen,qstart,qend,empty,target,tlen,tstart,tend,nident,alnlen,bits,cigar,fident,raw`,
            vbp
        )

        vb && log(vbp, "parse paf file")


        # log("parse paf file")
        K = open(read_mmseqs2, "$dir/res.paf")

        vb && log(vbp, "paf file parsed.")

        K
    end

    vb && log(vbp, "temporary directory deleted.")

    for hit in hits
        # transform the cigar string in tuples of (len, char)
        hit.cigar = collect(uncigar(hit.cigar))
    end
    vb && log(vbp, "hit uncigar (N. hits = $(length(hits)))")

    return hits
end

end
