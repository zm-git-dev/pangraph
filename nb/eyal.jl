### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ fb195cf0-8062-11eb-3c99-fde4a4241d8b
begin
	using JLD2
	using LinearAlgebra
	using StatsBase, Statistics
	using Plots, JSServe, PlutoUI

	import WGLMakie, PyPlot
end

# ╔═╡ 949bde72-81d3-11eb-3e45-dd042b57b2bb
using LaTeXStrings

# ╔═╡ 0de243c4-8063-11eb-3580-136f64776460
Page()

# ╔═╡ 19d832ec-8063-11eb-39cf-b3f130287cc4
function makie(f::Function)
    scene = WGLMakie.Scene(resolution = (600, 400))
	f(scene)
   
    return scene
end

# ╔═╡ 9a89abe2-8062-11eb-2614-f7cb7a8fed05
function ingredients(path::String)
	# this is from the Julia source code (evalfile in base/loading.jl)
	# but with the modification that it returns the module instead of the last object
	name = Symbol(basename(path))
	m = Module(name)
	Core.eval(m,
        Expr(:toplevel,
             :(eval(x) = $(Expr(:core, :eval))($name, x)),
             :(include(x) = $(Expr(:top, :include))($name, x)),
             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
             :(include($path))))
	m
end

# ╔═╡ c96f10f0-8062-11eb-3e4e-2fb11d63ba86
DataIO = ingredients("../src/io.jl").DataIO

# ╔═╡ dedf4344-8137-11eb-3dd6-33edcac405d9
DataFilter = ingredients("../src/filter.jl").DataFilter

# ╔═╡ 0c8e9d0a-81cc-11eb-304f-8d4b03ed97b1
PointCloud = ingredients("../src/geo.jl").PointCloud

# ╔═╡ d3ca4d32-811e-11eb-2993-8bc55dd24808
FIGS = "/home/nolln/root/data/seqspace/eyal/figs"

# ╔═╡ 93bf354e-8124-11eb-3b5a-414a3603360f
SAVE = false

# ╔═╡ 6aa479a6-813a-11eb-0c97-353a3f53b9b2
gr(html_output_format=:png)

# ╔═╡ 1d7393d0-811f-11eb-349a-bba38050379e
SAMPLE = "sample_4"

# ╔═╡ 6fda1e64-8138-11eb-13ed-977e26b0e0cc
DATA_DIR = "/home/nolln/data/star/$SAMPLE"

# ╔═╡ ec2e7784-8062-11eb-16b4-3b009b7c4f60
RAW_INPUT = "$DATA_DIR/Solo.out/Gene/filtered/matrix.mtx"

# ╔═╡ 82119d96-8138-11eb-0138-1b6e3b959ae2
JLD_INPUT = "$DATA_DIR/glm_fit.jld2"

# ╔═╡ de1e1208-8062-11eb-0ee0-a9db551ffcd0
scrna = open(RAW_INPUT) do io
	DataIO.read_mtx(io)
end;

# ╔═╡ 84185814-806d-11eb-28f7-fd158f7638bb
μ = vec(mean(scrna,dims=2)); n = vec(sum(scrna.>0,dims=2))

# ╔═╡ d208a304-806d-11eb-15d0-ab7366d88511
begin
	PyPlot.clf()
	PyPlot.hexbin(log10.(μ), log10.(n.+1), 
		gridsize=(30,30), 
		cmap="inferno"
	) 
	PyPlot.xlabel("log avg count/cell")
	PyPlot.ylabel("log number of expressing cells")
	if SAVE
		PyPlot.savefig("$FIGS/qc/$SAMPLE/count_vs_expressing_cells.png")
	end
	PyPlot.gcf()
end

# ╔═╡ 33a625ba-806f-11eb-1a68-bfadc0cc660b
begin
	h = histogram(log10.(μ),bins=100,legend=false) 
	vline!([-3]) 
	xlabel!("avg log count/per cell")
	if SAVE
		savefig("$FIGS/qc/$SAMPLE/count_per_cell_histogram.png")
	end
	h
end

# ╔═╡ 597a17d0-8063-11eb-3738-3793619ecd8c
data = DataFilter.genes(scrna);

# ╔═╡ 86f746e0-8074-11eb-23b3-972e81ea370a
md"""
## Data filtering
We need to scale the data so so that highly expressed genes and lowly expressed genes are equally weighted - i.e. focus on the variation between them not overall scale.
This is clear after looking at the principal components: dominated by inter-gene variation.

Define ``x_{in}`` to be the expression count for gene i and cell n

###### Idea 1: 
Naively, we just want to fit each gene to a given parameterized distribution independently.
Genes can then be re-expressed
However, we want to regress against our degrees of freedom, the total expression for each cell ``g_n \equiv \sum_i x_{in}``, to deal with systematic variations in sequencing depth.
We assume that the expression of gene i takes the form:

`` x_{in} = \text{exp}\left[\alpha_i + \beta_i \text{log}(g_n)\right] + \epsilon_{in} ``

Where ``\epsilon_{in}`` is a noise parameter with zero mean. Then, given a distribution ``\rho`` parameterized by it's mean we could then write the log likelihood as

`` \ell(\vec{\beta}_i) = \sum_n \rho(x_{in}|\mu_i = \text{exp}\left[\alpha_{i} + \beta_{i} \text{log}(g_n)\right]) ``

Thus for example, if we think the data is Poisson distributed, the above relation reduces to

`` \ell(\alpha_i, \beta_i) = \sum_n x_{in}\left(\alpha_i + \beta_i \text{log}(g_n)\right) - \text{exp}\left[\alpha_i + \beta_i \text{log}(g_n) \right] ``

Conversely, if we want to model an overdispersed counting distribution, we can turn to the negative binomial (with mean``=\mu`` and variance``=\mu + \mu^2 / \Theta``):

`` \ell(\alpha_i, \beta_i, \gamma_i) = \sum_n \text{log}(\Gamma(x_{in} +  \gamma_i)) - \text{log}(\Gamma(x_{in})) - \text{log}(\Gamma( \gamma_i)) +  \gamma_i\text{log}\left(\frac{\mu}{\mu +  \gamma_i}\right) + x_{in}\text{log}\left(\frac{\gamma_i}{\mu + \gamma_i}\right) ``

This procedure introduces ``3`` parameters per gene and thus will have to fit ``3n_g`` number of parameters in total.
Hafemeister, C. & Satija, R. (2019) claims this overfits and thus put an heurestic prior on the parameters.
Let's test this
"""

# ╔═╡ 91c38718-8138-11eb-0f18-85916a6aa8ca
@load JLD_INPUT fits

# ╔═╡ 24263ca8-80fe-11eb-0692-07698ae10d02


# ╔═╡ 21c52cf2-813b-11eb-1453-557d5b272ca1
M = vec(maximum(data,dims=2))

# ╔═╡ ba8c6388-8116-11eb-35d2-df884093be9f
begin
	E = map(fits) do r
		r.likelihood
	end
	α = map(fits) do r
		r.parameters[1]
	end
	β = map(fits) do r
		r.parameters[2]
	end
	γ = map(fits) do r
		r.parameters[3]
	end
	
	δα = map(fits) do r
		sqrt(abs(r.uncertainty[1]))
	end
	δβ = map(fits) do r
		sqrt(abs(r.uncertainty[2]))
	end
	δγ = map(fits) do r
		sqrt(abs(r.uncertainty[3]))
	end
	χ = vec(mean(data,dims=2))[1:length(E)]
end

# ╔═╡ d7070fe2-8116-11eb-35c0-496f1cad3008
begin
	l1 = scatter(χ,E,
		yscale=:log,
		xscale=:log, 
		alpha=0.1, 
		legend=false, 
		xlabel="average count/cell", 
		ylabel="average negative log likelihood /cell"
	)
	if SAVE
		savefig("$FIGS/qc/$SAMPLE/count_per_cell_vs_likelihood.png")
	end
	l1
end

# ╔═╡ 2f377f1c-8119-11eb-10a8-71fc4394927c
begin
	l2 = scatter(χ, α, marker_z=log10.(δα), 
		xscale=:log10, 
		alpha=0.1, 
		label="", 
		xlabel="average count/cell", 
		ylabel="estimated α", 
		colorbar_title="log uncertainty"
	)
	if SAVE
		savefig("$FIGS/qc/$SAMPLE/count_per_cell_vs_α.png")
	end
	l2
end

# ╔═╡ f9824c96-8119-11eb-3adb-c3cbc2986947
begin
	l3 = scatter(χ, β, 
		marker_z=log10.(δβ), 
		alpha=0.1, 
		xscale=:log10, 
		label="", 
		ylims=(-5,5), 
		xlabel="average count/cell", 
		ylabel="estimated β", 
		colorbar_title="log uncertainty"
	)
	if SAVE
		savefig("$FIGS/qc/$SAMPLE/count_per_cell_vs_β.png")
	end
	l3
end

# ╔═╡ 08a8018e-811a-11eb-0ce4-6711063fce7d
begin
	l4 = scatter(χ, γ, 
		marker_z=log10.(δγ), 
		label="", 
		alpha=0.1, 
		yscale=:log10, 
		xscale=:log10, 
		xlabel="average count/cell", 
		ylabel="estimated γ", 
		colorbar_title="log uncertainty"
	)
	if SAVE
		savefig("$FIGS/qc/$SAMPLE/count_per_cell_vs_γ.png")
	end
	l4
end

# ╔═╡ c81b6f9a-813a-11eb-00d1-1d92878cb882
begin
	PyPlot.clf()
	PyPlot.hexbin(log10.(χ), log10.(γ), 
		gridsize=(40,40), 
		cmap="inferno"
	) 
	PyPlot.xlabel("log avg count/cell")
	PyPlot.ylabel("log estimated γ")
	if SAVE
		PyPlot.savefig("$FIGS/qc/$SAMPLE/count_per_cell_vs_γ_hexbin.png")
	end
	PyPlot.gcf()
end

# ╔═╡ 0e619b6e-813b-11eb-188c-b7ad7b1c6e3a
begin
	l5 = scatter(χ, M, 
		marker_z=log10.(γ), 
		label="", 
		alpha=0.1, 
		yscale=:log10, 
		xscale=:log10, 
		xlabel="average count/cell", 
		ylabel="max count/cell", 
		colorbar_title="log uncertainty"
	)
	if SAVE
		savefig("$FIGS/qc/$SAMPLE/max_count_per_cell_vs_γ.png")
	end
	l5
end

# ╔═╡ dfe3851e-8064-11eb-049a-4dfed6e9ff3c
function null(data)
	ñ = zeros(size(data))
	for i in 1:size(data,1)
		sample!(data[i,:], view(ñ,i,:))
	end
	return ñ
end

# ╔═╡ 2d358d88-814b-11eb-0c11-23089a86b856
norm = vcat((fit.residuals' for (i,fit) ∈ enumerate(fits) if M[i] > 1 )...)

# ╔═╡ 18a61724-8156-11eb-07db-3de6b3104a1c
function gene_names()
	names = open("$DATA_DIR/Solo.out/Gene/filtered/features.tsv") do io
		[split(line)[2] for line in eachline(io)]
	end
	# NOTE: many details here depend strongly on how we filter...
	#		should just make scrna matrix into a struct so names are always bound
	@assert length(names) == size(scrna,1)

	μ 	  = vec(mean(scrna,dims=2))
	names = names[μ .> 1e-3]
	names = names[M .> 1]
	
	@assert length(names) == size(norm,1)

	return names
end

# ╔═╡ 8d4fa794-814b-11eb-2a06-1ba01e048084
F = svd(norm);

# ╔═╡ ded15dc2-814b-11eb-35d5-63bf4d616d34
F̃ = svd(null(norm));

# ╔═╡ d7a04a06-814b-11eb-0f00-49c8745fc905
begin
	p1 = plot(F.S/sum(F.S), xscale=:log10, yscale=:log10)
	plot!(F̃.S/sum(F̃.S), xscale=:log10, yscale=:log10)
	xlabel!("principal component")
	ylabel!("singular value")
	if SAVE
		savefig("$FIGS/geo/$SAMPLE/residual_matrix_singular_values.png")
	end
	p1
end

# ╔═╡ 8b400948-814c-11eb-059d-f1fa06164139
ipr(ψ) = sum(ψ.^4, dims=1) ./ (sum(ψ.^2, dims=1)).^2

# ╔═╡ cb23cd18-814c-11eb-3314-674bcab45578
ψ = ipr(F.U)

# ╔═╡ 075494be-81d6-11eb-123e-d3f005c9cb0d
localized_state, localized_size = getindex(argmin(1 ./ ψ),2), minimum(1 ./ ψ)

# ╔═╡ d50cb378-814c-11eb-1863-db97f0895d64
begin
	p2 = plot(1 ./ ψ', yscale=:log10, xscale=:log10)
	xlabel!("component")
	ylabel!("participation ratio")
	if SAVE
		savefig("$FIGS/geo/$SAMPLE/residual_matrix_participation_ratio.png")
	end
	p2
end

# ╔═╡ eb4592e0-8156-11eb-2564-49ede281d83a
names = gene_names()

# ╔═╡ 36a6f78a-81dc-11eb-187f-e91cdaa1f611
localized_genes = names[sortperm(F.U[:,localized_state].^2; rev=true)]

# ╔═╡ fff6ffa0-8156-11eb-0a6b-195479c7b49b
names[argmax(abs.(F.U[:,22]))]

# ╔═╡ 0a30bc3e-814d-11eb-3a4c-77164eb08043
begin
	plot(abs.(F.U[:,1]),yscale=:log10,alpha=.5)
	plot!(abs.(F.U[:,20]),alpha=.5)
end

# ╔═╡ 22a82bd4-81cb-11eb-0c85-af09449b8b9e
dₑ = 50; Ñ = F.U[:,1:dₑ] * Diagonal(F.S[1:dₑ]) * F.Vt[1:dₑ,:]

# ╔═╡ acc97a86-81df-11eb-39bd-6b7c3e599e1c
dd = 500; approx = F.U[:,1:dd] * Diagonal(F.S[1:dd]) * F.Vt[1:dd,:]

# ╔═╡ 6e7c5dc4-81de-11eb-1e29-6d94b51d76b8
let p
	p = plot(sqrt.(cumsum(F.S.^2)/sum(F.S.^2)), linewidth=2)
	xlabel!("rank approximation")
	ylabel!("correlation with original")
	
	if SAVE
		savefig("$FIGS/geo/$SAMPLE/svd_reduction_accuracy.png")
	end
	p
end

# ╔═╡ 40885c20-81dd-11eb-1b6d-733ca67aa312
ind_sample = sample(CartesianIndices(norm), 2000)

# ╔═╡ 574a4dd8-81dd-11eb-16a9-d9066f374f9d
scatter(approx[ind_sample], norm[ind_sample])

# ╔═╡ 0823bfd8-81cd-11eb-144d-9bb76ef0ea65
g̃ = PointCloud.geodesics(Ñ,12)

# ╔═╡ 69e3f8f8-81f7-11eb-3cc7-a7ac4ca669a5
if SAVE
	@save "$DATA_DIR/geodesic_k12.jld2" Ñ g̃
end

# ╔═╡ 4d0509e8-81cf-11eb-1e7b-8d8ad2ecf672
PyPlot.clf(); PyPlot.matshow(g̃); PyPlot.gcf()

# ╔═╡ ff7617a2-81cf-11eb-1c8f-c5dbacc6d897
ρ, Rs = PointCloud.scaling(g̃, 1000)

# ╔═╡ dce339b0-81d2-11eb-1f94-6906505d3346
begin
	p3 = plot(Rs, ρ', alpha=0.03, color=:cyan3, label="")
	plot!(Rs, mean(ρ, dims=1)', linewidth=5, alpha=1, color=:cyan3, label="mean")
	plot!(Rs, 1e-3*Rs.^3, linestyle=:dashdot, linewidth=2, alpha=1, color=:coral3, label=L"\alpha R^3")
	plot!(Rs, 5e-2*Rs.^2, linestyle=:dashdot, linewidth=2, alpha=1, color=:darkgoldenrod1, label=L"\alpha R^2")

	xaxis!("radius", :log10, (10, 400))
	yaxis!("number of points", :log10, (10, 3200))
	
	if SAVE
		savefig("$FIGS/geo/$SAMPLE/ball_scaling.png")
	end
	p3
end

# ╔═╡ 4e9e490e-81d4-11eb-2948-6f1b312907fc
ξ = PointCloud.mds(g̃.^2, 10)

# ╔═╡ 708ef216-81d4-11eb-3673-e394d0fa118f
begin

residual = zeros(size(ξ,2))
for d ∈ 1:size(ξ,2)
	D̄ = .√PointCloud.distance²(ξ[:,1:d]')
	
	residual[d] = cor(
		PointCloud.upper_tri(D̄), 
		PointCloud.upper_tri(g̃)
	)
end

end;

# ╔═╡ 97052e74-81d4-11eb-115f-0b6b19bea33a
begin
	p4 = plot(residual, linewidth=2, label="")
	xlabel!("dimension")
	ylabel!("correlation of distance matrices")
	if SAVE
		savefig("$FIGS/geo/$SAMPLE/isomap_distance_correlation.png")
	end
	p4
end

# ╔═╡ eb79c294-81d4-11eb-2cd6-633742a3ca0e
gene_index = Dict(g => i for (i,g) in enumerate(names))

# ╔═╡ 91e387a0-81da-11eb-1c9f-857e7079fe59
@bind THETA PlutoUI.Slider(range(0,180,length=100))

# ╔═╡ 9bb40ce6-81da-11eb-3814-670fc5029879
@bind PHI PlutoUI.Slider(range(0,360,length=100))

# ╔═╡ 449c5422-81e9-11eb-293d-2b3f8db9e449
GENE = "SOX11"

# ╔═╡ a9989792-81d9-11eb-2123-632f4c14bcd0
begin
	PyPlot.clf()
	PyPlot.scatter3D(ξ[:,1], ξ[:,2], ξ[:,3],
		c=vec(Ñ[gene_index[GENE],:]),
		cmap="inferno",
	)
	PyPlot.view_init(THETA,PHI)
	if false
		PyPlot.savefig("$FIGS/geo/$SAMPLE/isomap_coordinates_$GENE.png")
	end
	PyPlot.gcf()
end

# ╔═╡ 736505e4-81d6-11eb-0ed5-db36e1f6a6ec
#=WGLMakie.scatter(ξ[:,1], ξ[:,2], ξ[:,3], 
	color=vec(Ñ[gene_index["PAX6"],:]), 
	markersize=5
)=#

# ╔═╡ Cell order:
# ╠═fb195cf0-8062-11eb-3c99-fde4a4241d8b
# ╠═949bde72-81d3-11eb-3e45-dd042b57b2bb
# ╠═0de243c4-8063-11eb-3580-136f64776460
# ╠═19d832ec-8063-11eb-39cf-b3f130287cc4
# ╠═9a89abe2-8062-11eb-2614-f7cb7a8fed05
# ╠═c96f10f0-8062-11eb-3e4e-2fb11d63ba86
# ╠═dedf4344-8137-11eb-3dd6-33edcac405d9
# ╠═0c8e9d0a-81cc-11eb-304f-8d4b03ed97b1
# ╠═d3ca4d32-811e-11eb-2993-8bc55dd24808
# ╠═93bf354e-8124-11eb-3b5a-414a3603360f
# ╠═6aa479a6-813a-11eb-0c97-353a3f53b9b2
# ╠═1d7393d0-811f-11eb-349a-bba38050379e
# ╟─6fda1e64-8138-11eb-13ed-977e26b0e0cc
# ╟─ec2e7784-8062-11eb-16b4-3b009b7c4f60
# ╟─82119d96-8138-11eb-0138-1b6e3b959ae2
# ╟─de1e1208-8062-11eb-0ee0-a9db551ffcd0
# ╠═84185814-806d-11eb-28f7-fd158f7638bb
# ╟─d208a304-806d-11eb-15d0-ab7366d88511
# ╟─33a625ba-806f-11eb-1a68-bfadc0cc660b
# ╠═597a17d0-8063-11eb-3738-3793619ecd8c
# ╠═18a61724-8156-11eb-07db-3de6b3104a1c
# ╟─86f746e0-8074-11eb-23b3-972e81ea370a
# ╠═91c38718-8138-11eb-0f18-85916a6aa8ca
# ╟─24263ca8-80fe-11eb-0692-07698ae10d02
# ╟─21c52cf2-813b-11eb-1453-557d5b272ca1
# ╟─ba8c6388-8116-11eb-35d2-df884093be9f
# ╟─d7070fe2-8116-11eb-35c0-496f1cad3008
# ╟─2f377f1c-8119-11eb-10a8-71fc4394927c
# ╟─f9824c96-8119-11eb-3adb-c3cbc2986947
# ╟─08a8018e-811a-11eb-0ce4-6711063fce7d
# ╟─c81b6f9a-813a-11eb-00d1-1d92878cb882
# ╟─0e619b6e-813b-11eb-188c-b7ad7b1c6e3a
# ╠═dfe3851e-8064-11eb-049a-4dfed6e9ff3c
# ╠═2d358d88-814b-11eb-0c11-23089a86b856
# ╠═8d4fa794-814b-11eb-2a06-1ba01e048084
# ╠═ded15dc2-814b-11eb-35d5-63bf4d616d34
# ╠═d7a04a06-814b-11eb-0f00-49c8745fc905
# ╠═8b400948-814c-11eb-059d-f1fa06164139
# ╠═cb23cd18-814c-11eb-3314-674bcab45578
# ╠═075494be-81d6-11eb-123e-d3f005c9cb0d
# ╠═36a6f78a-81dc-11eb-187f-e91cdaa1f611
# ╠═d50cb378-814c-11eb-1863-db97f0895d64
# ╟─eb4592e0-8156-11eb-2564-49ede281d83a
# ╠═fff6ffa0-8156-11eb-0a6b-195479c7b49b
# ╠═0a30bc3e-814d-11eb-3a4c-77164eb08043
# ╠═22a82bd4-81cb-11eb-0c85-af09449b8b9e
# ╠═acc97a86-81df-11eb-39bd-6b7c3e599e1c
# ╠═6e7c5dc4-81de-11eb-1e29-6d94b51d76b8
# ╠═40885c20-81dd-11eb-1b6d-733ca67aa312
# ╠═574a4dd8-81dd-11eb-16a9-d9066f374f9d
# ╠═0823bfd8-81cd-11eb-144d-9bb76ef0ea65
# ╠═69e3f8f8-81f7-11eb-3cc7-a7ac4ca669a5
# ╠═4d0509e8-81cf-11eb-1e7b-8d8ad2ecf672
# ╠═ff7617a2-81cf-11eb-1c8f-c5dbacc6d897
# ╠═dce339b0-81d2-11eb-1f94-6906505d3346
# ╠═4e9e490e-81d4-11eb-2948-6f1b312907fc
# ╠═708ef216-81d4-11eb-3673-e394d0fa118f
# ╠═97052e74-81d4-11eb-115f-0b6b19bea33a
# ╠═eb79c294-81d4-11eb-2cd6-633742a3ca0e
# ╠═91e387a0-81da-11eb-1c9f-857e7079fe59
# ╠═9bb40ce6-81da-11eb-3814-670fc5029879
# ╠═449c5422-81e9-11eb-293d-2b3f8db9e449
# ╠═a9989792-81d9-11eb-2123-632f4c14bcd0
# ╠═736505e4-81d6-11eb-0ed5-db36e1f6a6ec
