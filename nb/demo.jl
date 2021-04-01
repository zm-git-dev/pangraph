### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ b5007056-88c2-11eb-3239-470733674ca3
using JLD2, FileIO

# ╔═╡ bea8d4fe-88c4-11eb-2a35-0f3276b7076d
begin
	import WGLMakie
	using JSServe, PlutoUI
end

# ╔═╡ 2bec13b6-88cb-11eb-1457-d95c48ff85a2
using Statistics

# ╔═╡ 8c7ee592-88d2-11eb-2789-512bf260207a
using StatsBase

# ╔═╡ ef87ed26-88db-11eb-3394-8fb21920d78d
using Plots

# ╔═╡ 8bcb8340-88c7-11eb-23ba-dd0227976b38
Page()

# ╔═╡ b1277040-88c4-11eb-2521-dfd292c1a1e0
function makie(f::Function)
    scene = WGLMakie.Scene(resolution = (600, 400))
	f(scene)
   
    return scene
end

# ╔═╡ b541d858-88c6-11eb-1e7d-13b6602f09ea
begin
	Core.eval(Main, :(include("../src/scrna.jl")))
	Core.eval(Main, :(include("../src/geo.jl")))
	Core.eval(Main, :(include("../src/infer.jl")))
end

# ╔═╡ dedf10d6-88c6-11eb-0ea2-514f6361094a
data = load("../3_4_5_6_7_my_normalization.jld2");

# ╔═╡ b701d1fa-88c3-11eb-1f24-bb1005bec7ab
begin
	seq = data["seq"]
	fit = data["fit"]
	embryo = data["embryo"]
	G = data["G"]
	D = data["D"]
	ξ = data["ξ"]
	ι = data["ι"]
	N = data["N"]
	F = data["F"]
	Ñ = data["Ñ"]
	r⃗ = data["r⃗"]
	ϕ = data["ϕ"]
	Ψ = data["Ψ"]
	Ψᵣ = data["Ψᵣ"]
	Ψₗ = data["Ψₗ"]
end;

# ╔═╡ ba24dc84-88cc-11eb-105d-97c13fdde18e
cor(Main.PointCloud.distance(ξ[:,1:3]')[:],  D[ι,ι][:])

# ╔═╡ 1250df04-88cb-11eb-1f48-959d13fc78d3
seq.gene[sortperm([corspearman(vec(N[i,ι]), ξ[:,1]) for i in 1:size(seq,1)])][1:50]

# ╔═╡ 40ec2332-88cb-11eb-2b86-6bcd9a437bad
seq.gene[sortperm([cor(vec(N[i,ι]), ξ[:,1]) for i in 1:size(seq,1)])][end-25:end]

# ╔═╡ 51172176-88cb-11eb-0899-1380eb084dac
seq.gene[sortperm([cor(vec(N[i,ι]), ξ[:,2]) for i in 1:size(seq,1)])][1:25]

# ╔═╡ 6218aee0-88cb-11eb-3d3a-497601ba793f
seq.gene[sortperm([cor(vec(N[i,ι]), ξ[:,2]) for i in 1:size(seq,1)])][end-25:end]

# ╔═╡ 71fb7928-88cb-11eb-1862-0be3d30b3e1a
seq.gene[sortperm([cor(vec(N[i,ι]), ξ[:,3]) for i in 1:size(seq,1)])][1:50]

# ╔═╡ 7aa9a540-88cb-11eb-0da4-e9aecdbf0967
seq.gene[sortperm([corspearman(vec(N[i,ι]), ξ[:,3]) for i in 1:size(seq,1)])][end-50:end]

# ╔═╡ 4797eeda-88c8-11eb-2402-23e143690c43
begin
	AP = r⃗[ι,1]
	DV = atan.(r⃗[ι,2],  r⃗[ι,3])
end;

# ╔═╡ 54205940-88c7-11eb-1b18-857742525b2f
makie() do s
	WGLMakie.scatter!(s, ξ[1:2:end,1], ξ[1:2:end,2], ξ[1:2:end,3], color=AP[1:2:end], markersize=2000)
end

# ╔═╡ 79f38d1c-88c8-11eb-20b3-877aba8b565b
makie() do s
	WGLMakie.scatter!(s, ξ[:,1], ξ[:,2], ξ[:,3], color=DV, markersize=2000)
end

# ╔═╡ b59d8c1c-88ca-11eb-0c32-edab0455d918
makie() do s
	WGLMakie.scatter!(s, ξ[:,1], ξ[:,2], ξ[:,3], 
		color=vec(N["shg",ι]), 
		markersize=2000
	)
end

# ╔═╡ 2769a0ee-88c9-11eb-1a77-63ab8dfa82e6
begin
   Ψ2  = ϕ(0.05)
   Ψₗ2 = Ψ2  ./ sum(Ψ,dims=2)
   Ψᵣ2 = (Ψ2 ./ sum(Ψ,dims=1))'

   G2  = Main.scRNA.Count(seq.data * Ψₗ2',seq.gene,fill("",3039))

   r⃗2 = Ψᵣ2 * embryo
end;

# ╔═╡ 06670396-88d1-11eb-3817-7db908739dd3
mt_index = occursin.("mt:", G2.gene)

# ╔═╡ 97d25ed0-88c8-11eb-1fea-fb8a3bcbb95e
makie() do s
	WGLMakie.scatter!(s, -embryo[:,1], embryo[:,2], embryo[:,3], 
		color=G2["sna",:], #vec(sum(G2[mt_index,:],dims=1)),
		markersize=2000)
end

# ╔═╡ e69357f6-88e4-11eb-39f7-d514156f22a5
makie() do s
	WGLMakie.scatter!(s, -embryo[:,1], embryo[:,2], embryo[:,3], 
		color=G2["myo",:], #vec(sum(G2[mt_index,:],dims=1)),
		markersize=2000)
end

# ╔═╡ fd3e3ace-88db-11eb-1376-f5a1e24b25f3
scatter(embryo[:,1], vec(G2["run",:])); scatter!(embryo[:,1], vec(G2["eve",:]).+.10); scatter!(embryo[:,1], vec(G2["ftz",:]).-.1);xaxis!((-100, 100))

# ╔═╡ Cell order:
# ╠═b5007056-88c2-11eb-3239-470733674ca3
# ╠═bea8d4fe-88c4-11eb-2a35-0f3276b7076d
# ╠═2bec13b6-88cb-11eb-1457-d95c48ff85a2
# ╠═8c7ee592-88d2-11eb-2789-512bf260207a
# ╠═8bcb8340-88c7-11eb-23ba-dd0227976b38
# ╠═b1277040-88c4-11eb-2521-dfd292c1a1e0
# ╠═b541d858-88c6-11eb-1e7d-13b6602f09ea
# ╠═dedf10d6-88c6-11eb-0ea2-514f6361094a
# ╟─b701d1fa-88c3-11eb-1f24-bb1005bec7ab
# ╠═ba24dc84-88cc-11eb-105d-97c13fdde18e
# ╠═1250df04-88cb-11eb-1f48-959d13fc78d3
# ╟─40ec2332-88cb-11eb-2b86-6bcd9a437bad
# ╟─51172176-88cb-11eb-0899-1380eb084dac
# ╟─6218aee0-88cb-11eb-3d3a-497601ba793f
# ╠═71fb7928-88cb-11eb-1862-0be3d30b3e1a
# ╠═7aa9a540-88cb-11eb-0da4-e9aecdbf0967
# ╠═4797eeda-88c8-11eb-2402-23e143690c43
# ╠═54205940-88c7-11eb-1b18-857742525b2f
# ╠═79f38d1c-88c8-11eb-20b3-877aba8b565b
# ╠═b59d8c1c-88ca-11eb-0c32-edab0455d918
# ╠═06670396-88d1-11eb-3817-7db908739dd3
# ╠═97d25ed0-88c8-11eb-1fea-fb8a3bcbb95e
# ╠═e69357f6-88e4-11eb-39f7-d514156f22a5
# ╠═ef87ed26-88db-11eb-3394-8fb21920d78d
# ╠═fd3e3ace-88db-11eb-1376-f5a1e24b25f3
# ╟─2769a0ee-88c9-11eb-1a77-63ab8dfa82e6
