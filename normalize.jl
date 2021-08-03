using JLD2, FileIO
using LinearAlgebra, SpecialFunctions, NMF
using Distributions, Statistics, StatsBase
using Plots

include("src/scrna.jl")
using .scRNA

include("src/filter.jl")
using .DataFilter

include("src/util.jl")
using .Utility

include("src/geo.jl")
using .PointCloud

# ------------------------------------------------------------------------
# printing functions

alert(msg) = println(stderr, msg)
function userinput(msg, default=nothing)
    type = typeof(default)
    if default !== nothing
        print(stdout, "$msg [default=$(default)]: ")
    else
        print(stdout, msg)
    end
    input = readline(stdin)
    return length(input) > 0 ? parse(typeof(default), input) : default
end

# ------------------------------------------------------------------------
# plotting functions

module Plot

using Statistics, StatsBase
using Plots, ColorSchemes

using ..DataFilter

rank(x) = invperm(sortperm(x)) / length(x)

cdf(x; kwargs...)  = plot(sort(x),  range(0,1,length=length(x)); kwargs...)
cdf!(x; kwargs...) = plot!(sort(x), range(0,1,length=length(x)); kwargs...)

function countmarginals(data; ϵ=1e-6)
	p₁ = cdf(vec(mean(data,dims=1)).+ϵ, xscale=:log10, label="", linewidth=2)
	xaxis!("mean count/cell")
	yaxis!("CDF")

	p₂ = cdf(vec(mean(data,dims=2)).+ϵ, xscale=:log10, label="", linewidth=2)
	xaxis!("mean count/gene")
	yaxis!("CDF")

    return plot(p₁, p₂)
end

function qq(γ)
	logγ = log.(γ)
	
	model = MLE.generalized_normal(logγ)
	param = MLE.fit(model)
	
	p = scatter(rank(logγ), model.cumulative(param), linewidth=2, label="")
	plot!(0:1, 0:1, color=:red, linewidth=2, linestyle=:dashdot, label="ideal")
	xaxis!("empirical quantile")
	yaxis!("model quantile")

	return p
end

function genefits(X, p)
	mₓ = vec(mean(X, dims=2))
	Mₓ = vec(maximum(X, dims=2));

	p₁ = scatter(mₓ, p.α, 
		alpha=0.1, 
		xscale=:log10, 
		marker_z=log10.(Mₓ), 
		label=false
	)
	xaxis!("expression / cell")
	yaxis!("estimated α")

	p₂ = scatter(mₓ, p.β, 
		alpha=0.1, 
		xscale=:log10, 
		marker_z=log10.(Mₓ), 
		label=false
	)
	xaxis!("expression / cell")
	yaxis!("estimated β")

	p₃ = scatter(mₓ, p.γ, 
		alpha=0.1, 
		xscale=:log10,
		yscale=:log10, 
		marker_z=log10.(Mₓ), 
		label=false
	)
	xaxis!("expression / cell")
	yaxis!("estimated γ")

    return p₁, p₂, p₃
end

function mpmaxeigval(X, λ; ϵ=2)
    p = plot(λ, 1:length(λ), 
             xscale=:log10, 
             yscale=:log10, 
             linewidth=2, 
             label="Empirical distribution"
    )
	
	k = sum(λ .> (sqrt(size(X,1))+sqrt(size(X,2))) - ϵ)

	vline!([(sqrt(size(X,1))+sqrt(size(X,2))-ϵ)], linewidth=2, linestyle=:dashdot, label="MP maximum eigenvalue")
	title!("scRNAseq data (k=$k)")
	xaxis!("singular value")
	yaxis!("CDF")
	
	p
end

end

# ------------------------------------------------------------------------
# processing functions

function filterdrosophila(data)
    markers  = (
        yolk = scRNA.locus(data, "sisA", "CG8129", "Corp", "CG8195", "CNT1", "ZnT77C"),
        pole = scRNA.locus(data, "pgc"),
        dvir = scRNA.searchloci(data, "Dvir_")
    )

    data = scRNA.filtercell(data) do cell, _
        (sum(cell[markers.yolk]) < 10 
      && sum(cell[markers.pole]) < 3 
      && sum(cell[markers.dvir]) < .25*sum(cell))
    end

    scRNA.filtergene(data) do _, gene
        !occursin("Dvir_", gene)
    end
end

function rawdata(root::AbstractString; replicates=missing)
    X = if ismissing(replicates)
        scRNA.load(root)
    else
        reduce(∪, scRNA.load("$root/$d") for d ∈ readdir(root) if occursin(replicates,d))
    end
    alert("--> raw dimensions of count matrix: $(size(X))")

    # XXX: hacky
    if occursin("drosophila",root)
        alert("--> filtering...")
        X = filterdrosophila(X)
        alert("--> raw dimensions of filtered count matrix: $(size(X))")
    end

    return X
end

function filterdata(x, cutoff)
	x = scRNA.filtergene(x) do gene, _
		mean(gene) >= cutoff.gene && length(unique(gene)) > 3
	end
	
	x = scRNA.filtercell(x) do cell, _
		mean(cell) >= cutoff.cell
	end

    alert("--> dimensions of filtered count matrix: $(size(x))")
    return x
end

function fitdata(X, y)
    logy = log.(y)

    model = MLE.generalized_normal(logy)
    param = MLE.fit(model)

    X, p = MLE.fit_glm(:negative_binomial, X; Γ=(β̄=1, δβ¯²=10, Γᵧ=param))

    return p, X
end

# ------------------------------------------------------------------------
# main point of entry

function main(root, showplot::Bool; replicates=missing)
    alert("> loading raw data...")
    X = rawdata(root; replicates=replicates)
    p = Plot.countmarginals(X)
    savefig(p, "$root/figs/raw_count_marginals.png")
    cutoff = if showplot
        display(plot(p))
        (gene=userinput("--> cutoff for genes", 5e-3),
         cell=userinput("--> cutoff for cells", 0.5))
    else
        (gene=5e-3, cell=0.5)
    end

    alert("> filtering raw data...")
    X = filterdata(X, cutoff)
    p = Plot.countmarginals(X)
    savefig(p, "$root/figs/filtered_count_marginals.png")
    if showplot
        display(p)
        userinput("--> press enter to continue") 
    end

    alert("> estimating empirical gene overdispersion distribution...")
    _, p₀ = MLE.fit_glm(:negative_binomial, X; 
        Γ=(β̄=1, δβ¯²=10, Γᵧ=nothing),
        run=(x) -> mean(x) > 1			
    );

	p = Plot.cdf(p₀.γ, xlabel="estimated γ", ylabel="CDF", xscale=:log10, label="", linewidth=2)
    savefig(p, "$root/figs/nb_γ_empirical_distribution.png")
    if showplot
        display(p)
        userinput("--> press enter to continue") 
    end

    p = Plot.qq(p₀.γ)
    savefig(p, "$root/figs/nb_γ_prior_fit.png")
	if showplot
        display(p)
        userinput("--> press enter to continue") 
	end

    alert("> estimating posterior gene overdispersion distribution...")
    p₁, ρ = fitdata(X, p₀.γ)
    save("$root/proc/fit_nb.jld2", Dict("data"=>X.data, "gene"=>X.gene, "cell"=>X.cell, "p₁"=>p₁, "p₀"=>p₀, "residual"=>ρ))

    p = Plot.genefits(X, p₁)
    savefig(p[1], "$root/figs/nb_α_fit.png")
    savefig(p[2], "$root/figs/nb_β_fit.png")
    savefig(p[3], "$root/figs/nb_γ_fit.png")
	if showplot
        display(plot(p..., layout=(1,3)))
        userinput("--> press enter to continue") 
	end

    alert("> estimating normalized count variance...")
    X̃, Ṽ, u², v² = let
        σ² = X.*(X.+p₁.γ) ./ (1 .+ p₁.γ)
        u², v², _ = Utility.sinkhorn(σ²)

        (Diagonal(.√u²) * X * Diagonal(.√v²)), (Diagonal(u²) * σ² * Diagonal(v²)), u², v²
    end;
    save("$root/proc/normalized_raw_count.jld2", Dict("X̃"=>X̃, "Ṽ"=>Ṽ, "u²"=>u², "v²"=>v², "gene"=>X.gene, "cell"=>X.cell))

    alert("> estimating rank...")
    p = Plot.mpmaxeigval(X̃, svdvals(X̃))
	dim = if showplot
        display(p)
        userinput("--> desired rank", 100)
    else
        50
	end
    savefig(p, "$root/figs/estimated_rank.png")

    alert("> estimating mean count matrix...")
    Y = let d = dim 
        r = nnmf(X̃, d; alg=:cd)
        r.W*r.H
    end 
    alert("--> variance captured by dimensional reduction: $(cor(Y[:], X̃[:]))")

    Ỹ, u, v = let
        u, v, _ = Utility.sinkhorn(Y)
        (Diagonal(u) * Y * Diagonal(v)), u, v
    end
    save("$root/proc/normalized_mean_count_rank_$(dim).jld2", Dict("Ỹ"=>Ỹ, "u"=>u, "v"=>v, "gene"=>X.gene, "cell"=>X.cell))
end

if abspath(PROGRAM_FILE) == @__FILE__
    root = ARGS[1]
    !isdir(root) && error("directory $root not found")
    if length(ARGS) == 1
        main(root, false)
    else
        main(root, false; replicates=ARGS[2])
    end
end
