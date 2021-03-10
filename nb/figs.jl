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

# ╔═╡ 489b13da-7d23-11eb-39de-57a12469721f
using Plots, LaTeXStrings

# ╔═╡ 82c5386a-7d2f-11eb-3992-e985ee16933c
using ColorSchemes

# ╔═╡ 4bd45848-7d23-11eb-1a8b-b36f7461f368
begin
	import WGLMakie
	using JSServe, PlutoUI
end

# ╔═╡ fa6e4914-7d24-11eb-0ff2-1974e4ec8df2
using StatsBase, Statistics, LinearAlgebra

# ╔═╡ 2ef712f2-7d2e-11eb-228c-2d023dc6ab10
using Random

# ╔═╡ a1b684a0-7d40-11eb-2a4c-e990f12c6cbb
using Distributions

# ╔═╡ 4256e210-7d23-11eb-2f7a-1d8aceab0682
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

# ╔═╡ 3d2bf622-7d23-11eb-33fb-3f8a4f0f309e
function makie(f::Function)
    scene = WGLMakie.Scene(resolution = (600, 400))
	f(scene)
   
    return scene
end

# ╔═╡ 5e1fa2de-7d23-11eb-18da-ddf0ec5714cc
M = ingredients("../src/main.jl")

# ╔═╡ a0c3a118-7d3b-11eb-0309-a9540887541f
I = ingredients("../src/infer.jl")

# ╔═╡ fce24760-7d22-11eb-2474-a5efdd176182
scrna, genes = M.SeqSpace.expression();

# ╔═╡ 330fd846-7d25-11eb-3933-b5f8871eedc5
size(scrna)

# ╔═╡ 4ddffd5a-7d24-11eb-2c70-675569787f53
"""
This section generates figures for linear dimensional reduction
"""

# ╔═╡ 72a6c262-7d25-11eb-1a05-756c421933f5
pvals(matrix) = (λ=reverse(eigvals(cov(matrix,dims=2)))[1:1297])/sum(λ)

# ╔═╡ 85650856-7d24-11eb-055d-75a99e549182
function sample_from!(null, matrix)
	@assert size(null) == size(matrix)
	for g ∈ 1:size(null,1)
		sample!(matrix[g,:], view(null,g,:); replace=false)
	end
end

# ╔═╡ c3fdc5ce-7d26-11eb-235a-33e432adcd91
function null_pvals(matrix; niter=10)
	null = zeros(size(matrix))
	λ 	 = zeros(minimum(size(matrix)), niter)
	for i ∈ 1:niter
		sample_from!(null, matrix)
		λ[:,i] = pvals(null)
	end
	
	return λ
end

# ╔═╡ 7bf6e4e4-7d27-11eb-0d11-a343a5ead1fe
λ = (
	data = pvals(scrna),
	null = null_pvals(scrna),
);

# ╔═╡ 564474de-7d2a-11eb-107a-094fc5b153f6
λ.null

# ╔═╡ de3c3612-7d25-11eb-3233-393941f590e5
begin
	plot(λ.data[1:end-1], label="data", color=:cyan3, linewidth=3)
	for i ∈ 1:size(λ.null,2)
		plot!(λ.null[1:end-1,i], label="", color=:orangered2, alpha=0.5, linewidth=1)
	end
	plot!(λ.null[1:end-1,1], label="null", color=:orangered2, alpha=1, linewidth=1)

	xaxis!("principal value", :log10)
	yaxis!("fraction of variance", :log10)
	
	#savefig("../figs/principal_values.svg")
end

# ╔═╡ 59156b62-7d2c-11eb-1f14-3139307e312e
"""
This section generates figures analyzing state localization
"""

# ╔═╡ 713cebde-7d2c-11eb-3918-31d392278150
ipr(ψ) = sum(ψ.^4, dims=1) ./ (sum(ψ.^2, dims=1)).^2

# ╔═╡ 151d45be-7d2d-11eb-00ae-495dff2916f2
function scan_ipr(x, Ns; niter=5)
	IPR = zeros(size(x,2), niter, length(Ns))
	for (n,N) ∈ enumerate(Ns)
		for i ∈ 1:niter
			ι = randperm(size(x,1))[1:N]
			IPR[:,i,n] = ipr(x[ι,:])
		end
	end
	IPR
end

# ╔═╡ f940b7ee-7d2d-11eb-3ecf-930c0eb1b938
F = svd(scrna);

# ╔═╡ 1ab88ad2-7d2e-11eb-2327-3f2015c5220c
Ns = [100,250,500,750,1000,2000,3000,4000,5000,6000,7000,8000,size(F.U,1)]; IPR = scan_ipr(F.U, Ns; niter=50);

# ╔═╡ 476449e0-7d2e-11eb-2d91-11e5a358d060
ĪP̄R̄ = dropdims(mean(IPR,dims=2);dims=2)

# ╔═╡ 4a089ff2-7d2e-11eb-0e3b-9d580d220c45
begin
	cmap = ColorSchemes.plasma
	plot(Ns, 1 ./ ĪP̄R̄[1,:],color=get(cmap,0.0),linewidth=2,label="")
	for i in 2:49
		plot!(Ns, 1 ./ ĪP̄R̄[i,:],color=get(cmap,(i-1)/50),linewidth=2,label="")
	end
	plot!(Ns, 1 ./ ĪP̄R̄[50,:],color=get(cmap,1.0),linewidth=2,label="")
	
	xaxis!("number of genes subsampled", :log10)
	yaxis!(L"1 / \sum \psi^4", :log10)
	
	#savefig("../figs/participation_ratio_subsample_scaling.svg")
end

# ╔═╡ 49dc2956-7d31-11eb-03c8-f5e29d9d0798
ψ = ipr(F.U)

# ╔═╡ 58a0f57a-7d31-11eb-3b59-293c95b4aa07
begin
	plot(1 ./ ψ')
	
	xaxis!("principal component", :log10)
	yaxis!("participation ratio", :log10)
	
	#savefig("../figs/participation_ratio.svg")
end

# ╔═╡ b499f6bc-7d34-11eb-106d-53126cd7a5c6
"""
This section generates figures analyzing point cloud scaling.
"""

# ╔═╡ c115efae-7d34-11eb-0d58-bba15e9922bb
G = F.U[:, 1:25] * Diagonal(F.S[1:25]) * F.Vt[1:25,:];

# ╔═╡ 1a663bc4-7d35-11eb-20e2-99f298281450
Dₑ = M.SeqSpace.PointCloud.distance(G);

# ╔═╡ 3838406e-7d35-11eb-31fb-256eea2a72cb
Dₚ = M.SeqSpace.PointCloud.geodesics(G, 6);

# ╔═╡ 629180aa-7d35-11eb-2bd8-47deefc64f91
function scaling(D, N)
	Rₘᵢₙ = minimum(D[D .> 0])
	Rₘₐₓ = maximum(D)
	Rs   = range(Rₘᵢₙ,Rₘₐₓ,length=N)
	
	ϕ = zeros(size(D,1), N)
	for (i,R) ∈ enumerate(Rs)
		ϕ[:,i] = sum(D .<= R, dims=1)
	end
	
	return ϕ, Rs
end

# ╔═╡ 4c497cfc-7d36-11eb-013a-07b8d183909f
ρₑ, Rₑ = scaling(Dₑ, 1000);

# ╔═╡ f3da8bec-7d37-11eb-06c3-e58f9312c561
ρₚ, Rₚ = scaling(Dₚ, 1000);

# ╔═╡ 61689ce4-7d36-11eb-206f-a362c5f1b24b
begin
	plot(Rₑ, ρₑ', alpha=0.03, color=:cyan3, label="", title="Euclidean")
	plot!(Rₑ, mean(ρₑ, dims=1)', linewidth=5, alpha=1, color=:cyan3, label="mean")
	plot!(Rₑ, 5e-10*Rₑ.^7, linestyle=:dashdot, linewidth=2, alpha=1, color=:black, label=L"\alpha R^7")

	xaxis!("radius", :log10, (25, 400))
	yaxis!("number of points", :log10, (10, 1297))
	
	#savefig("../figs/euclidean_ball_scaling.svg")
end

# ╔═╡ f1e88e9a-7d37-11eb-0529-a3ab42f90864
begin
	plot(Rₚ, ρₚ', alpha=0.02, color=:cyan3, label="", title="Geodesic")
	plot!(Rₚ, mean(ρₚ, dims=1)', linewidth=5, alpha=1, color=:cyan3, label="mean")
	plot!(Rₚ, 9e-5*Rₚ.^3.2, linestyle=:dashdot, linewidth=2, alpha=1, color=:black, label=L"\alpha R^3")

	xaxis!("radius", :log10, (25, 400))
	yaxis!("number of points", :log10, (10, 1297))
	
	#savefig("../figs/geodesic_ball_scaling.svg")
end

# ╔═╡ e2a9cb1c-7d3a-11eb-088a-7d51b807d714
"""
This section generates figures analyzing isomap
"""

# ╔═╡ 8c321e78-7d3b-11eb-375b-5dd081921104
invert, embryo = I.Inference.inversion(); Ψ = invert(0.5);

# ╔═╡ a7c62f6e-7d3b-11eb-1ef0-f3743d0e58e1
begin
	ψᵣ = Ψ  ./ sum(Ψ, dims=2)
	ψₗ = (Ψ ./ sum(Ψ, dims=1))'
end;

# ╔═╡ b0bc877e-7d3b-11eb-1110-913567d545da
r̂ = (ψₗ * embryo)'; AP = r̂[1,:]; DV = atan.(r̂[2,:], r̂[3,:])

# ╔═╡ ed76e2e6-7d3a-11eb-04ed-69f2bdd9f68a
ξ = M.SeqSpace.PointCloud.mds(Dₚ.^2, 10)

# ╔═╡ 0c2a0f80-7d3e-11eb-17b6-71185c47896d
begin

residual = zeros(size(ξ,2))
for d ∈ 1:size(ξ,2)
	D̄ = .√M.SeqSpace.PointCloud.distance²(ξ[:,1:d]')
	
	residual[d] = cor(M.SeqSpace.PointCloud.upper_tri(D̄), M.SeqSpace.PointCloud.upper_tri(Dₚ))
end

end;

# ╔═╡ 694eb1e4-7d42-11eb-0ac6-8fe4f3896971
begin
	d̄ = M.SeqSpace.PointCloud.upper_tri(.√M.SeqSpace.PointCloud.distance²(ξ[:,1:6]'));
	d = M.SeqSpace.PointCloud.upper_tri(Dₚ)
	
	ι = sample(1:length(d), 5000)
	
	scatter(d[ι], d̄[ι], legend=false)
	
	xaxis!("estimated geodesic distance")
	yaxis!("pairwise euclidean distance in latent space")
	#savefig("../figs/distance_isomap_correlation_eg_d3.svg")
end

# ╔═╡ 8cc4e066-7d3e-11eb-37b7-312b8d2a4a73
begin
	plot(residual, linewidth=2, label="")

	xaxis!("latent dimension")
	yaxis!("correlation of distance matrices", (0.7,0.95))
	
	#savefig("../figs/distance_correlation_vs_isomap_dimension.svg")
end

# ╔═╡ 43b9ec00-7d3d-11eb-2973-cbb9af2197a8
@bind THETA PlutoUI.Slider(range(0,90,length=100))

# ╔═╡ 6cca0e84-7d3d-11eb-1a00-77473b588de7
@bind PHI PlutoUI.Slider(range(0,90,length=100))

# ╔═╡ ffdcaae6-7d3b-11eb-3ddb-558d8e1638d6
begin
	scatter(ξ[:,2],-ξ[:,1],ξ[:,3],
		marker_z=AP,
		markersize=5,
		xlims=(-150,150),
		ylims=(-150,150),
		zlims=(-125,75),
		title="predicted AP", 
		camera=(THETA,PHI))
	
	#savefig("../figs/isomap_3d_AP.svg")
end

# ╔═╡ a3fd873e-7d3d-11eb-391e-cfc5d3c80797
begin
	scatter(ξ[:,2],-ξ[:,1],ξ[:,3],
		marker_z=DV,
		markersize=5,
		xlims=(-150,150),
		ylims=(-150,150),
		zlims=(-125,75),
		title="predicted DV", 
		camera=(THETA,PHI))
	
	#savefig("../figs/isomap_3d_DV.svg")
end

# ╔═╡ 66270b84-7d46-11eb-2e09-c97d61f6425b
begin
scatter(F.Vt[3,:],F.Vt[4,:],F.Vt[5,:],
		marker_z=AP,
		markersize=5,
		title="predicted AP", 
		camera=(THETA,PHI)
)
	
#savefig("../figs/svd_3d_AP.svg")

end

# ╔═╡ 927cd290-7d46-11eb-057c-c14cffce4ce5
begin
	
scatter(F.Vt[2,:],F.Vt[3,:],F.Vt[4,:],
		marker_z=DV,
		markersize=5,
		title="predicted DV", 
		camera=(THETA,PHI)
)

#savefig("../figs/svd_3d_DV.svg")

end

# ╔═╡ d1c09536-7d46-11eb-2d39-79a39782580b
Dₓ = M.SeqSpace.PointCloud.geodesics(r̂,6)

# ╔═╡ ead3da76-7d46-11eb-0d7c-cb4496b2c27a
begin
	dx = M.SeqSpace.PointCloud.upper_tri(Dₓ)
	dp = M.SeqSpace.PointCloud.upper_tri(Dₚ)
	
	ι₂ = sample(1:length(dx), 5000)

	scatter(dx[ι₂], dp[ι₂])
	xlabel!("spatial geodesic distance")
	ylabel!("expression geodesic distance")
	
	#savefig("../figs/spatial_vs_expression_geodesic.svg")
end

# ╔═╡ Cell order:
# ╠═489b13da-7d23-11eb-39de-57a12469721f
# ╠═82c5386a-7d2f-11eb-3992-e985ee16933c
# ╠═4bd45848-7d23-11eb-1a8b-b36f7461f368
# ╠═4256e210-7d23-11eb-2f7a-1d8aceab0682
# ╠═3d2bf622-7d23-11eb-33fb-3f8a4f0f309e
# ╠═5e1fa2de-7d23-11eb-18da-ddf0ec5714cc
# ╠═a0c3a118-7d3b-11eb-0309-a9540887541f
# ╠═fce24760-7d22-11eb-2474-a5efdd176182
# ╠═330fd846-7d25-11eb-3933-b5f8871eedc5
# ╠═4ddffd5a-7d24-11eb-2c70-675569787f53
# ╠═fa6e4914-7d24-11eb-0ff2-1974e4ec8df2
# ╠═72a6c262-7d25-11eb-1a05-756c421933f5
# ╠═85650856-7d24-11eb-055d-75a99e549182
# ╠═c3fdc5ce-7d26-11eb-235a-33e432adcd91
# ╠═7bf6e4e4-7d27-11eb-0d11-a343a5ead1fe
# ╠═564474de-7d2a-11eb-107a-094fc5b153f6
# ╠═de3c3612-7d25-11eb-3233-393941f590e5
# ╠═59156b62-7d2c-11eb-1f14-3139307e312e
# ╠═713cebde-7d2c-11eb-3918-31d392278150
# ╠═151d45be-7d2d-11eb-00ae-495dff2916f2
# ╠═f940b7ee-7d2d-11eb-3ecf-930c0eb1b938
# ╠═2ef712f2-7d2e-11eb-228c-2d023dc6ab10
# ╠═1ab88ad2-7d2e-11eb-2327-3f2015c5220c
# ╠═476449e0-7d2e-11eb-2d91-11e5a358d060
# ╠═4a089ff2-7d2e-11eb-0e3b-9d580d220c45
# ╠═49dc2956-7d31-11eb-03c8-f5e29d9d0798
# ╠═58a0f57a-7d31-11eb-3b59-293c95b4aa07
# ╠═b499f6bc-7d34-11eb-106d-53126cd7a5c6
# ╠═c115efae-7d34-11eb-0d58-bba15e9922bb
# ╠═1a663bc4-7d35-11eb-20e2-99f298281450
# ╠═3838406e-7d35-11eb-31fb-256eea2a72cb
# ╠═629180aa-7d35-11eb-2bd8-47deefc64f91
# ╠═4c497cfc-7d36-11eb-013a-07b8d183909f
# ╠═f3da8bec-7d37-11eb-06c3-e58f9312c561
# ╠═61689ce4-7d36-11eb-206f-a362c5f1b24b
# ╠═f1e88e9a-7d37-11eb-0529-a3ab42f90864
# ╠═e2a9cb1c-7d3a-11eb-088a-7d51b807d714
# ╠═8c321e78-7d3b-11eb-375b-5dd081921104
# ╠═a7c62f6e-7d3b-11eb-1ef0-f3743d0e58e1
# ╠═b0bc877e-7d3b-11eb-1110-913567d545da
# ╠═ed76e2e6-7d3a-11eb-04ed-69f2bdd9f68a
# ╠═a1b684a0-7d40-11eb-2a4c-e990f12c6cbb
# ╠═0c2a0f80-7d3e-11eb-17b6-71185c47896d
# ╠═694eb1e4-7d42-11eb-0ac6-8fe4f3896971
# ╠═8cc4e066-7d3e-11eb-37b7-312b8d2a4a73
# ╠═43b9ec00-7d3d-11eb-2973-cbb9af2197a8
# ╠═6cca0e84-7d3d-11eb-1a00-77473b588de7
# ╠═ffdcaae6-7d3b-11eb-3ddb-558d8e1638d6
# ╠═a3fd873e-7d3d-11eb-391e-cfc5d3c80797
# ╠═66270b84-7d46-11eb-2e09-c97d61f6425b
# ╠═927cd290-7d46-11eb-057c-c14cffce4ce5
# ╠═d1c09536-7d46-11eb-2d39-79a39782580b
# ╠═ead3da76-7d46-11eb-0d7c-cb4496b2c27a
