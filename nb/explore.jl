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

# ╔═╡ 4514f886-7d6c-11eb-2c69-27e88d82d6c9
using StatsBase, Statistics

# ╔═╡ 57f7faf2-7d53-11eb-193e-298f1b811802
using PlutoUI

# ╔═╡ 56695996-7b8d-11eb-1612-07fab5dafac3
import Plots

# ╔═╡ a4a14288-7b90-11eb-01db-9dba870c417e


# ╔═╡ c408a1fe-7b90-11eb-391f-23c775403ddf


# ╔═╡ c81b443e-7b90-11eb-1500-372fb43712b3


# ╔═╡ f90a2afc-7b8a-11eb-31ed-f79ca667aed0
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

# ╔═╡ 1abf567c-7b8b-11eb-1cd5-176f3985cf5f
M = ingredients("../src/main.jl")

# ╔═╡ af24807a-7b96-11eb-031b-a554e04fd797
I = ingredients("../src/infer.jl")

# ╔═╡ 36f6f034-7b8b-11eb-0f7c-3bf426e8c858
param = M.SeqSpace.HyperParams(;
	N  = 500, 
	Ws = [100,50,50,50,50,25],
	BN = [1,2,3,4],
	DO = [1],
	V  = 145,
	B  = 128,
	kₗ = 5,
	δ  = 1,
	γₓ = 5e-4,
	γₛ = 0,
	η  = 1e-3,
	dₒ = 3,
)

# ╔═╡ 4ce81e9a-7b8b-11eb-2335-c960ad189017
result, data = M.SeqSpace.run(param); r = result[1]

# ╔═╡ 54f7b29c-7b8d-11eb-3fe4-f5d4ee676431
begin
	Plots.plot(r.loss.train, linewidth=3, label="train"); 
	Plots.plot!(r.loss.valid, linewidth=3, label="validate")
	#Plots.savefig("~/root/src/seqspace/figs/scrna_loss.svg")
end

# ╔═╡ d8066234-7b90-11eb-331e-9549202425c1
begin
	z = r.model.pullback(data.x)
	x̂ = r.model.pushforward(z)
end;

# ╔═╡ 77980d76-7ba4-11eb-2f5c-3f55f1540e8d
begin
	Plots.scatter(data.x[1,:],x̂[1,:],alpha=.5,legend=false) 
	for i ∈ 2:8 
		Plots.scatter!(data.x[i,:], x̂[i,:],alpha=.2) 
	end;
	Plots.xlabel!("original SVD coordinate")
	Plots.ylabel!("reconstructed SVD coordinate")

	#Plots.savefig("~/root/src/seqspace/figs/scrna_svd_reconstructed_vs_measured.png")

	#Plots.scatter!(data.x[10,:],x̂[10,:],alpha=.2)
end

# ╔═╡ 02e14eae-7bac-11eb-17ed-43043c6eac4f
mean(x) = sum(x)/length(x)

# ╔═╡ 085eb4dc-7b97-11eb-02c1-1761f9ed4e8c
invert, embryo = I.Inference.inversion(); Ψ = invert(0.5);

# ╔═╡ dbe1bdac-7b9c-11eb-0e3a-577e252a1f78
begin
	ψᵣ = Ψ  ./ sum(Ψ, dims=2)
	ψₗ = (Ψ ./ sum(Ψ, dims=1))'
end;

# ╔═╡ 9b6544b0-7b9c-11eb-13aa-bf2ce2abeb55
r̂ = (ψₗ * embryo)'; AP = r̂[1,:]; DV = atan.(r̂[2,:], r̂[3,:])

# ╔═╡ fdedda4a-7b90-11eb-30ba-d9c76a5ccd08
#=
makie() do s
	scatter!(s, z[1,:], z[2,:], z[3,:], color=AP, markersize=500)
end
=#

# ╔═╡ ae5ab5ca-7bc2-11eb-0dee-fb999487b907
scrna, genes = M.SeqSpace.expression(); scrnaᵣ, λ, Φ = M.SeqSpace.ML.preprocess(scrna; dₒ=35);

# ╔═╡ 24c252f4-7d6c-11eb-34f0-09a2da3ab676
begin
	SI = sample(CartesianIndices(scrna), 5000)
	XX = scrna[SI]
	YY = data.map(x̂)[SI]
	Plots.scatter(XX[XX .>0], YY[XX .> 0], label="correlation≈.83")
	Plots.xlabel!("original scrnaseq count")
	Plots.ylabel!("estimated scrnaseq count")
	#Plots.savefig("~/root/src/seqspace/figs/scrna_genes_vs_reconstructed.png")
end

# ╔═╡ a0c401f4-7d6c-11eb-3a29-af05e369934f
cor(XX[XX .>0], YY[XX .> 0])

# ╔═╡ 5c390788-7bc2-11eb-1a73-391f5cbbb17e
z̄ = M.SeqSpace.PointCloud.isomap(Φ(scrnaᵣ), 3; sparse=true);

# ╔═╡ c6feb4f4-7bc3-11eb-25b5-f778fb6d0621
#=
makie() do s
	scatter!(s, z̄[:,1], z̄[:,2], z̄[:,3], color=AP, markersize=3000)
end
=#

# ╔═╡ 8a6f7c50-7d53-11eb-0866-4f13c3f89e08
Plots.plotly()

# ╔═╡ 508da442-7d53-11eb-02f3-0196dec17b84
@bind THETA Slider(range(0,90,length=100))

# ╔═╡ 541b0e60-7d53-11eb-194a-d99f76f26465
@bind PHI Slider(range(0,90,length=100))

# ╔═╡ 64274152-7d53-11eb-0e2f-0f6dceb7b4fb
begin
	Plots.scatter(ψᵣ*z[1,:], ψᵣ*z[2,:], ψᵣ*z[3,:],
		markersize=2,
		marker_z=embryo[:,1],
		camera=(THETA,PHI),
		legend=false
	)
	#Plots.savefig("~/root/src/seqspace/figs/neural_network_AP_eg2.html")
end

# ╔═╡ f236e1da-7d55-11eb-1d90-1329298ffc7b
begin
	Plots.scatter(ψᵣ*z[1,:], ψᵣ*z[2,:], ψᵣ*z[3,:],
		markersize=2,
		marker_z=atan.(embryo[:,2],embryo[:,3]),
		camera=(THETA,PHI),
		legend=false
	)
	#Plots.savefig("~/root/src/seqspace/figs/neural_network_DV_eg2.html")
end

# ╔═╡ Cell order:
# ╠═56695996-7b8d-11eb-1612-07fab5dafac3
# ╟─a4a14288-7b90-11eb-01db-9dba870c417e
# ╟─c408a1fe-7b90-11eb-391f-23c775403ddf
# ╟─c81b443e-7b90-11eb-1500-372fb43712b3
# ╠═f90a2afc-7b8a-11eb-31ed-f79ca667aed0
# ╠═1abf567c-7b8b-11eb-1cd5-176f3985cf5f
# ╠═af24807a-7b96-11eb-031b-a554e04fd797
# ╠═36f6f034-7b8b-11eb-0f7c-3bf426e8c858
# ╠═4ce81e9a-7b8b-11eb-2335-c960ad189017
# ╠═54f7b29c-7b8d-11eb-3fe4-f5d4ee676431
# ╠═d8066234-7b90-11eb-331e-9549202425c1
# ╠═77980d76-7ba4-11eb-2f5c-3f55f1540e8d
# ╠═4514f886-7d6c-11eb-2c69-27e88d82d6c9
# ╠═24c252f4-7d6c-11eb-34f0-09a2da3ab676
# ╠═a0c401f4-7d6c-11eb-3a29-af05e369934f
# ╠═02e14eae-7bac-11eb-17ed-43043c6eac4f
# ╠═085eb4dc-7b97-11eb-02c1-1761f9ed4e8c
# ╠═dbe1bdac-7b9c-11eb-0e3a-577e252a1f78
# ╠═9b6544b0-7b9c-11eb-13aa-bf2ce2abeb55
# ╠═fdedda4a-7b90-11eb-30ba-d9c76a5ccd08
# ╠═ae5ab5ca-7bc2-11eb-0dee-fb999487b907
# ╠═5c390788-7bc2-11eb-1a73-391f5cbbb17e
# ╠═c6feb4f4-7bc3-11eb-25b5-f778fb6d0621
# ╠═57f7faf2-7d53-11eb-193e-298f1b811802
# ╠═8a6f7c50-7d53-11eb-0866-4f13c3f89e08
# ╠═508da442-7d53-11eb-02f3-0196dec17b84
# ╠═541b0e60-7d53-11eb-194a-d99f76f26465
# ╠═64274152-7d53-11eb-0e2f-0f6dceb7b4fb
# ╠═f236e1da-7d55-11eb-1d90-1329298ffc7b
