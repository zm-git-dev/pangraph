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

# ╔═╡ 3bfe8b4c-81f2-11eb-122d-bd6d98d3dc4f
begin
	using JLD2
	using LinearAlgebra
	using StatsBase, Statistics
	using Plots, JSServe, PlutoUI
	using LaTeXStrings
	
	import WGLMakie, PyPlot
end

# ╔═╡ 28d30d2c-81f2-11eb-3c67-a7383ad853de
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

# ╔═╡ 55ba2d7a-81f2-11eb-0afd-bb4b2ad4db16
DataIO = ingredients("../src/io.jl").DataIO

# ╔═╡ 586bdd98-81f2-11eb-10de-5d67fd7c28f3
DataFilter = ingredients("../src/filter.jl").DataFilter

# ╔═╡ 5bd3926e-81f2-11eb-3eac-497e0a9ff48a
PointCloud = ingredients("../src/geo.jl").PointCloud

# ╔═╡ 70e06bbe-81f2-11eb-0896-d3f932f19cda
ROOT = "/home/nolln/root/data/seqspace/raw"

# ╔═╡ 4e398d10-81f8-11eb-331c-0f430e248ac3
SAVE=false

# ╔═╡ 8d8b1974-81f2-11eb-2df8-45e329cfbe78
SAMPLE = "rep3"

# ╔═╡ 994c7886-81f2-11eb-24e4-edd6fece2098
gr(html_output_format=:png)

# ╔═╡ ab58264c-81f2-11eb-146c-ab082b41c4bb
DATA = "$ROOT/$SAMPLE"

# ╔═╡ 0fb03c38-81f3-11eb-19f3-4951c52e759a
FIGS = "$DATA/figs"

# ╔═╡ 1ee0ba70-81f3-11eb-1f36-d5ce11db0701
scrna = open("$DATA/matrix.mtx") do io
	DataIO.read_mtx(io)
end;

# ╔═╡ bc9dd25c-81f8-11eb-3ed6-cd12b63976d3
size(scrna)

# ╔═╡ 1c645a06-81f4-11eb-321f-ff025c62a777
UMI = vec(sum(scrna, dims=1))

# ╔═╡ 56a9e004-81f5-11eb-219c-cfe55c815269
mean(UMI), var(UMI)

# ╔═╡ d5091a98-81ff-11eb-143c-d120eafda3de
MIN_UMI = 1.1e3

# ╔═╡ e0663e36-8207-11eb-3307-59786cc96054
function filter_cells()
	highUMI = UMI .> MIN_UMI
	genes = open("$DATA/features.tsv") do io
		[split(line)[2] for line in eachline(io)]
	end
	
	pole_genes = Set(["pgc", "Eno"])
	pole_markers = map(genes) do g
		g ∈ pole_genes
	end
	
	notpole = vec(sum(scrna[pole_markers,:], dims=1)) .< 3
	
	dvir_markers = map(genes) do g
		occursin("Dvir_", g)
	end
	
	notdvir = vec(sum(scrna[dvir_markers,:], dims=1) ./ sum(scrna, dims=1)) .< .1
	
	return highUMI .& notpole .& notdvir
end

# ╔═╡ 37f75734-81f4-11eb-00fc-55a5d52fd30a
let p
	p = plot(sort(UMI;rev=true).+1, 
		yscale=:log10,
		linewidth=3
	)
	hline!([MIN_UMI])
	xlabel!("rank")
	ylabel!("total UMI")
	if SAVE
		savefig("$FIGS/total_umi_cell_rank.png")
	end
	p
end

# ╔═╡ 84c32904-81f8-11eb-38c5-8bf8ff24a47d
raw = scrna[:,filter_cells()];

# ╔═╡ b2f45fb4-81f8-11eb-3d33-73e39fb64ac3
size(raw)

# ╔═╡ d20c607c-81f8-11eb-2b58-f750686b84a8
μ = vec(mean(scrna,dims=2)); n = vec(sum(scrna.>0,dims=2))

# ╔═╡ f57aabe6-81fa-11eb-3e59-379c27c61375
CUTOFF = 5e-3

# ╔═╡ f0ec2fd6-81f8-11eb-261e-4d1f9ed523c6
let p
	p = histogram(log10.(μ),bins=100,legend=false) 
	vline!([log10(CUTOFF)]) 
	xlabel!("avg log count/per cell")
	if SAVE
		savefig("$FIGS/umi_count_per_cell_histogram.png")
	end
	p
end

# ╔═╡ 2cb9cb86-81f9-11eb-185f-dde76238f4cf
simple = DataFilter.genes(raw; min=CUTOFF);

# ╔═╡ 8eca5386-81f9-11eb-1b0a-256e1d04ff31
size(simple)

# ╔═╡ a37e0e60-81f9-11eb-2863-91bd4e2eef8b
fits = DataFilter.fit_glm(simple;σ¯²=1e-2, μ=1.0)

# ╔═╡ f0f1f23a-81f9-11eb-119b-7b0796efcddf
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
	χ = vec(mean(simple,dims=2))[1:length(E)]
end

# ╔═╡ fdbc491e-81f9-11eb-0651-abd1056d8d51
let p
	p = scatter(χ,E,
		yscale=:log,
		xscale=:log, 
		alpha=0.1, 
		legend=false, 
		xlabel="average count/cell", 
		ylabel="average negative log likelihood /cell"
	)
	if SAVE
		savefig("$FIGS/count_per_cell_vs_likelihood.png")
	end
	p
end

# ╔═╡ 194456ba-81fa-11eb-3780-4f6551e15330
let p
	p = scatter(χ, α, marker_z=log10.(δα), 
		xscale=:log10, 
		alpha=0.1, 
		label="", 
		xlabel="average count/cell", 
		ylabel="estimated α", 
		colorbar_title="log uncertainty"
	)
	if SAVE
		savefig("$FIGS/count_per_cell_vs_α.png")
	end
	p
end

# ╔═╡ 34510a5a-81fa-11eb-2417-8f9d8618004a
let p
	p = scatter(χ, β, 
		marker_z=log10.(δβ), 
		alpha=0.1, 
		xscale=:log10, 
		label="", 
		ylims=(0.5,1.5), 
		xlabel="average count/cell", 
		ylabel="estimated β", 
		colorbar_title="log uncertainty"
	)
	if SAVE
		savefig("$FIGS/count_per_cell_vs_β.png")
	end
	p
end

# ╔═╡ 8a7b7ec4-81fa-11eb-0cdc-71d9dabdafd3
let p
	p = scatter(χ, γ, 
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
		savefig("$FIGS/count_per_cell_vs_γ.png")
	end
	p
end

# ╔═╡ 717a8444-81fa-11eb-28e0-019fbe333fe4
begin
	PyPlot.clf()
	PyPlot.hexbin(log10.(χ), log10.(γ), 
		gridsize=(40,40), 
		cmap="inferno"
	) 
	PyPlot.xlabel("log avg count/cell")
	PyPlot.ylabel("log estimated γ")
	if SAVE
		PyPlot.savefig("$FIGS/count_per_cell_vs_γ_hexbin.png")
	end
	PyPlot.gcf()
end

# ╔═╡ bb8a7b2a-81fa-11eb-369d-09e6281405c3
M = vec(maximum(simple,dims=2))

# ╔═╡ a3961952-81fa-11eb-1ee0-0b99b18ed929
let p
	p = scatter(χ, M, 
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
		savefig("$FIGS/max_count_per_cell_vs_γ.png")
	end
	p
end

# ╔═╡ c915b6c4-81fa-11eb-0a73-9d5729670289
N = vcat((fit.residuals' for (i,fit) ∈ enumerate(fits) if M[i] > 1 )...)

# ╔═╡ e7e5b284-81fa-11eb-3077-c710544971a4
function gene_names()
	names = open("$DATA/features.tsv") do io
		[split(line)[2] for line in eachline(io)]
	end
	# NOTE: many details here depend strongly on how we filter...
	#		should just make scrna matrix into a struct so names are always bound
	@assert length(names) == size(raw,1)

	μ 	  = vec(mean(raw,dims=2))
	names = names[μ .> CUTOFF]
	
	@assert length(names) == length(M)
	
	names = names[M .> 1]
	
	@assert length(names) == size(N,1)

	return names
end

# ╔═╡ 27f7e770-81fb-11eb-25df-031089c32b6d
genes = gene_names(); index = Dict(gene=>i for (i,gene) in enumerate(genes))

# ╔═╡ 25422052-85ea-11eb-0682-4b20ec3d469d
begin
	dvir_markers = map(genes) do g
		occursin("Dvir_", g)
	end
		
	notdvir = vec(sum(scrna[dvir_markers,:], dims=1) ./ sum(scrna, dims=1))
end

# ╔═╡ 2e46507c-85ea-11eb-19d8-2d852be3eeaa
plot(sort(notdvir), range(0,1,length=length(notdvir)))

# ╔═╡ 1f6e008c-8215-11eb-0a43-cb3b28e76b13
let genes, p
	genes = open("$DATA/features.tsv") do io
		[split(line)[2] for line in eachline(io)]
	end
	mt_markers = map(genes) do g
		occursin("mt:", g)
	end
	
	frac_mt = vec(sum(scrna[mt_markers,:], dims=1) ./ sum(scrna, dims=1))
	p = plot(sort(frac_mt), range(0,1,length=length(frac_mt)),linewidth=2,label="")
	xlabel!("fraction of expression from mitochrondia")
	ylabel!("CDF")
	
	if SAVE
		savefig("$FIGS/mitochrondial_fraction.png")
	end
	p
end

# ╔═╡ 127bdc52-81fc-11eb-0edd-057847c839c9
F = svd(N)

# ╔═╡ 1824cfc4-81fc-11eb-016c-359492ff8bf0
let p
	p = plot(F.S/sum(F.S), yscale=:log10, xscale=:log10)
	xlabel!("component")
	ylabel!("singular value")
	if SAVE
		savefig("$FIGS/singular_values.png")
	end
	p
end

# ╔═╡ 861ca236-81fc-11eb-13a7-876888768422
let p
	p = plot(sqrt.(cumsum(F.S.^2)/sum(F.S.^2)), linewidth=2)
	xlabel!("rank approximation")
	ylabel!("correlation with original")
	
	if SAVE
		savefig("$FIGS/svd_reduction_correlation.png")
	end
	p
end

# ╔═╡ 5c11af0c-81fc-11eb-2472-2b6c20a6d55f
ipr(ψ) = sum(ψ.^4, dims=1) ./ (sum(ψ.^2, dims=1)).^2

# ╔═╡ 6043d57a-81fc-11eb-32c7-0525d51e5778
ψ = ipr(F.U)

# ╔═╡ 67490a68-81fc-11eb-1129-cb55a12f604e
let p
	p = plot(1 ./ ψ', yscale=:log10, xscale=:log10)
	xlabel!("component")
	ylabel!("participation ratio")
	if SAVE
		savefig("$FIGS/residual_matrix_participation_ratio.png")
	end
	p
end

# ╔═╡ b7419f9c-81fc-11eb-1f7c-c1d8ae3f04da
dₑ = 35; Ñ = F.U[:,1:dₑ] * Diagonal(F.S[1:dₑ]) * F.Vt[1:dₑ,:]; cor(Ñ[:], N[:])

# ╔═╡ e585ae0e-81fa-11eb-0ded-f388a80a268f
plot(sort(Ñ[index["hb"],:]), range(0,1,length=size(N,2)))

# ╔═╡ f162493a-85f6-11eb-367d-dfc988eff78b
scatter(N[index["hb"],:], Ñ[index["hb"],:])

# ╔═╡ b022e50e-81fc-11eb-2c8e-2f3e7a1071c3
g̃ = PointCloud.geodesics(Ñ,10);

# ╔═╡ e72b9ec4-81fc-11eb-06b5-0fcb2ef4b338
ρ, Rs = PointCloud.scaling(g̃, 1000);

# ╔═╡ fa2e15ba-81fc-11eb-3c9f-bba065692338
let p
	p = plot(Rs, ρ', alpha=0.03, color=:cyan3, label="")
	plot!(Rs, mean(ρ, dims=1)', linewidth=5, alpha=1, color=:cyan3, label="mean")
	plot!(Rs, 1e-2*Rs.^3, linestyle=:dashdot, linewidth=2, alpha=1, color=:coral3, label=L"\alpha R^3")
	plot!(Rs, 5e-1*Rs.^2, linestyle=:dashdot, linewidth=2, alpha=1, color=:darkgoldenrod1, label=L"\alpha R^2")

	xaxis!("radius", :log10, (5, 200))
	yaxis!("number of points", :log10, (10, 1200))
	
	if SAVE
		savefig("$FIGS/ball_scaling_dimensionality.png")
	end
	p
end

# ╔═╡ ecfb8e76-81fd-11eb-03b4-e356c6139365
ξ = PointCloud.mds(Hermitian(g̃.^2), 10)

# ╔═╡ f6bc2b00-81fd-11eb-33e5-37d6ee1f9368
let p
	residual = zeros(size(ξ,2))
	for d ∈ 1:size(ξ,2)
		D = .√PointCloud.distance²(ξ[:,1:d]')

		residual[d] = cor(
			PointCloud.upper_tri(D), 
			PointCloud.upper_tri(g̃)
		)
	end
	
	p = plot(
		residual, 
		linewidth=2, 
		label=""
	)
	
	xlabel!("dimension")
	ylabel!("correlation of distance matrices")
	if SAVE
		savefig("$FIGS/isomap_distance_correlation.png")
	end
	p
end

# ╔═╡ 896f2ed0-827a-11eb-304c-c38f1676693f
Inference = ingredients("../src/infer.jl").Inference

# ╔═╡ d1c8fcbe-8366-11eb-2b53-f35ebde4840e
ν,ω = jldopen("../static/params2.jld2") do fd
	fd["ν"], fd["ω"]
end

# ╔═╡ a07cc6f0-827a-11eb-3500-0b465b1ed0ab
Φ, embryo, database, match = Inference.inversion(N, genes;ν=ν,ω=ω);

# ╔═╡ 62d05d6e-85f7-11eb-1182-a1666eafff03
let genes
	genes = open("$DATA/features.tsv") do io
		[split(line)[2] for line in eachline(io)]
	end
	db_genes = Set(keys(database.gene))
	db_markers = map(genes) do g
		g ∈ db_genes
	end
	
	db_sum = vec(sum(scrna[db_markers,:], dims=1))
	plot(sort(db_sum.+1),range(0,1,length=length(db_sum)), xscale=:log10)
end

# ╔═╡ 31895884-827b-11eb-1edc-cd61ebd542c0
begin
	Ψ  = Φ(0.1)
	Ψₗ = Ψ  ./ sum(Ψ,dims=2)
	Ψᵣ = (Ψ ./ sum(Ψ,dims=1))'
	
	G  = N * Ψₗ'
	Gᵣ = G[match,:]
	
	r⃗ = Ψᵣ * embryo
end;

# ╔═╡ f865ae9e-827d-11eb-0bb2-cf83f10003ef
function scanβ()
	βs = range(0.01, 3, length=25)
	ρ  = zeros(84, length(βs))
	for (i,β) ∈ enumerate(βs)
		Ψ  = Φ(β)
		Ψₗ = Ψ  ./ sum(Ψ,dims=2)
		Ψᵣ = (Ψ ./ sum(Ψ,dims=1))'

		G  = N * Ψₗ'
		Gᵣ = G[match,:]
		for j ∈ 1:84
			ρ[j,i] = cor(Gᵣ[j,:], database.data[j,:])
		end
	end
	
	return βs, ρ
end

# ╔═╡ d2b8cbf6-827e-11eb-3fb8-450f699f44a9
begin
	Βs, CCs = scanβ()
	
	plot(Βs, CCs', alpha=0.25, color=:red, label="")
	plot!(Βs, mean(CCs',dims=2), ribbon=std(CCs',dims=2), fillalpha=0.25, color=:black, linewidth=2)
end

# ╔═╡ b87a436a-8365-11eb-3629-79a1945bec43
argmin(CCs[:,5])

# ╔═╡ 787d3fb4-827c-11eb-16db-e1e3c7975c1a
CC = [corspearman(Gᵣ[i,:], database.data[i,:]) for i in 1:size(Gᵣ,1)]

# ╔═╡ 174d2d8c-827c-11eb-38ed-079202945dcb
begin
	scatter(Gᵣ[1,:], database.data[1,:],label="",alpha=0.1)
	for i in 2:size(Gᵣ,1)
		scatter!(Gᵣ[i,:], database.data[i,:],label="",alpha=0.1)
	end
	xlabel!("estimated")
	ylabel!("database")
end

# ╔═╡ 47cffa4e-81fe-11eb-2068-39fbcc00e07c
GENE = "eve"

# ╔═╡ 3ce1b582-81fe-11eb-3840-f389f69bfc37
@bind THETA PlutoUI.Slider(range(-90,180,length=100))

# ╔═╡ 41b05e6a-81fe-11eb-0f93-416bb60edf9c
@bind PHI PlutoUI.Slider(range(0,360,length=100))

# ╔═╡ 4d854c64-81fe-11eb-0738-299577568527
begin
	PyPlot.clf()
	PyPlot.scatter3D(ξ[:,1], ξ[:,2],ξ[:,3],
		c=vec(N[index[GENE],:]),
		cmap="inferno",
		alpha=0.5,
	)
	PyPlot.view_init(THETA,PHI)
	if false
		PyPlot.savefig("$FIGS/isomap_coordinates_$GENE.png")
	end
	PyPlot.gcf()
end

# ╔═╡ 6b1a9a24-8201-11eb-2c4a-556b92432815
begin
	c1 = [corspearman(ξ[:,1], N[i,:]) for i in 1:size(Ñ,1)]
	c2 = [corspearman(ξ[:,2], N[i,:]) for i in 1:size(Ñ,1)]
	c3 = [corspearman(ξ[:,3], N[i,:]) for i in 1:size(Ñ,1)]
end

# ╔═╡ 90fcec56-8201-11eb-3c40-c735ca7cfe5a
begin
	SP1 = sortperm(c1;rev=true)
	SP2 = sortperm(c2;rev=true)
	SP3 = sortperm(c3;rev=true)
end

# ╔═╡ 88378d26-820e-11eb-3425-75ba2afb55b2
genes[SP1[1:50]]

# ╔═╡ 6ce4b132-821a-11eb-304b-bde9e2282434
genes[SP2[1:end]]

# ╔═╡ 6e4b9b1c-821a-11eb-26ca-5fa1b704de20
genes[SP3[1:50]]

# ╔═╡ 2f8e5b88-8280-11eb-01de-591fcb6b2cda
@bind THETA2 PlutoUI.Slider(range(-90,90,length=100))

# ╔═╡ 34d54e58-8280-11eb-1483-0179574a1c96
@bind PHI2 PlutoUI.Slider(range(0,180,length=100))

# ╔═╡ 120cc5a0-8365-11eb-2de3-31359c0fea1b
database.gene

# ╔═╡ 5a29a83e-85f7-11eb-1378-43e80893b73a
length(keys(database.gene))

# ╔═╡ 54f55324-8280-11eb-00d1-bbeb7b05501e
GENE2 = "run"

# ╔═╡ 38ff1874-8280-11eb-1846-a3db7627e4b5
begin
	PyPlot.clf()
	PyPlot.scatter3D(embryo[:,1], embryo[:,2], embryo[:,3],
		c=vec(G[index[GENE2],:]),
		cmap="inferno",
	)
	PyPlot.view_init(THETA2,PHI2)
	if false
		PyPlot.savefig("$FIGS/real_coordinates_est_$GENE.png")
	end
	PyPlot.gcf()
end

# ╔═╡ 53e6cc0c-85f7-11eb-0000-714188d3bb57


# ╔═╡ Cell order:
# ╠═3bfe8b4c-81f2-11eb-122d-bd6d98d3dc4f
# ╟─28d30d2c-81f2-11eb-3c67-a7383ad853de
# ╟─55ba2d7a-81f2-11eb-0afd-bb4b2ad4db16
# ╟─586bdd98-81f2-11eb-10de-5d67fd7c28f3
# ╟─5bd3926e-81f2-11eb-3eac-497e0a9ff48a
# ╟─70e06bbe-81f2-11eb-0896-d3f932f19cda
# ╟─4e398d10-81f8-11eb-331c-0f430e248ac3
# ╠═8d8b1974-81f2-11eb-2df8-45e329cfbe78
# ╟─994c7886-81f2-11eb-24e4-edd6fece2098
# ╟─ab58264c-81f2-11eb-146c-ab082b41c4bb
# ╟─0fb03c38-81f3-11eb-19f3-4951c52e759a
# ╟─1ee0ba70-81f3-11eb-1f36-d5ce11db0701
# ╠═bc9dd25c-81f8-11eb-3ed6-cd12b63976d3
# ╠═1c645a06-81f4-11eb-321f-ff025c62a777
# ╠═56a9e004-81f5-11eb-219c-cfe55c815269
# ╠═d5091a98-81ff-11eb-143c-d120eafda3de
# ╠═25422052-85ea-11eb-0682-4b20ec3d469d
# ╠═2e46507c-85ea-11eb-19d8-2d852be3eeaa
# ╠═62d05d6e-85f7-11eb-1182-a1666eafff03
# ╠═e0663e36-8207-11eb-3307-59786cc96054
# ╟─37f75734-81f4-11eb-00fc-55a5d52fd30a
# ╠═84c32904-81f8-11eb-38c5-8bf8ff24a47d
# ╠═1f6e008c-8215-11eb-0a43-cb3b28e76b13
# ╠═b2f45fb4-81f8-11eb-3d33-73e39fb64ac3
# ╠═d20c607c-81f8-11eb-2b58-f750686b84a8
# ╠═f57aabe6-81fa-11eb-3e59-379c27c61375
# ╟─f0ec2fd6-81f8-11eb-261e-4d1f9ed523c6
# ╠═2cb9cb86-81f9-11eb-185f-dde76238f4cf
# ╠═8eca5386-81f9-11eb-1b0a-256e1d04ff31
# ╠═a37e0e60-81f9-11eb-2863-91bd4e2eef8b
# ╠═f0f1f23a-81f9-11eb-119b-7b0796efcddf
# ╟─fdbc491e-81f9-11eb-0651-abd1056d8d51
# ╟─194456ba-81fa-11eb-3780-4f6551e15330
# ╠═34510a5a-81fa-11eb-2417-8f9d8618004a
# ╠═8a7b7ec4-81fa-11eb-0cdc-71d9dabdafd3
# ╟─717a8444-81fa-11eb-28e0-019fbe333fe4
# ╠═bb8a7b2a-81fa-11eb-369d-09e6281405c3
# ╠═a3961952-81fa-11eb-1ee0-0b99b18ed929
# ╟─e7e5b284-81fa-11eb-3077-c710544971a4
# ╠═27f7e770-81fb-11eb-25df-031089c32b6d
# ╠═c915b6c4-81fa-11eb-0a73-9d5729670289
# ╠═e585ae0e-81fa-11eb-0ded-f388a80a268f
# ╠═f162493a-85f6-11eb-367d-dfc988eff78b
# ╠═127bdc52-81fc-11eb-0edd-057847c839c9
# ╟─1824cfc4-81fc-11eb-016c-359492ff8bf0
# ╟─861ca236-81fc-11eb-13a7-876888768422
# ╠═5c11af0c-81fc-11eb-2472-2b6c20a6d55f
# ╠═6043d57a-81fc-11eb-32c7-0525d51e5778
# ╟─67490a68-81fc-11eb-1129-cb55a12f604e
# ╠═b7419f9c-81fc-11eb-1f7c-c1d8ae3f04da
# ╠═b022e50e-81fc-11eb-2c8e-2f3e7a1071c3
# ╠═e72b9ec4-81fc-11eb-06b5-0fcb2ef4b338
# ╟─fa2e15ba-81fc-11eb-3c9f-bba065692338
# ╠═ecfb8e76-81fd-11eb-03b4-e356c6139365
# ╟─f6bc2b00-81fd-11eb-33e5-37d6ee1f9368
# ╠═896f2ed0-827a-11eb-304c-c38f1676693f
# ╠═d1c8fcbe-8366-11eb-2b53-f35ebde4840e
# ╠═a07cc6f0-827a-11eb-3500-0b465b1ed0ab
# ╠═31895884-827b-11eb-1edc-cd61ebd542c0
# ╠═f865ae9e-827d-11eb-0bb2-cf83f10003ef
# ╠═d2b8cbf6-827e-11eb-3fb8-450f699f44a9
# ╠═b87a436a-8365-11eb-3629-79a1945bec43
# ╠═787d3fb4-827c-11eb-16db-e1e3c7975c1a
# ╠═174d2d8c-827c-11eb-38ed-079202945dcb
# ╠═47cffa4e-81fe-11eb-2068-39fbcc00e07c
# ╠═3ce1b582-81fe-11eb-3840-f389f69bfc37
# ╠═41b05e6a-81fe-11eb-0f93-416bb60edf9c
# ╠═4d854c64-81fe-11eb-0738-299577568527
# ╠═6b1a9a24-8201-11eb-2c4a-556b92432815
# ╠═90fcec56-8201-11eb-3c40-c735ca7cfe5a
# ╠═88378d26-820e-11eb-3425-75ba2afb55b2
# ╠═6ce4b132-821a-11eb-304b-bde9e2282434
# ╠═6e4b9b1c-821a-11eb-26ca-5fa1b704de20
# ╠═2f8e5b88-8280-11eb-01de-591fcb6b2cda
# ╠═34d54e58-8280-11eb-1483-0179574a1c96
# ╠═120cc5a0-8365-11eb-2de3-31359c0fea1b
# ╠═5a29a83e-85f7-11eb-1378-43e80893b73a
# ╠═54f55324-8280-11eb-00d1-bbeb7b05501e
# ╠═38ff1874-8280-11eb-1846-a3db7627e4b5
# ╠═53e6cc0c-85f7-11eb-0000-714188d3bb57
