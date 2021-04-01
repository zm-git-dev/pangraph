### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 2466c2a4-90c7-11eb-3f9c-5b87a7a35bb6
begin
	using LinearAlgebra, Statistics, StatsBase
	using NMF
	using Plots, ColorSchemes
end

# ╔═╡ 969b3f50-90bb-11eb-2b67-c784d20c0eb2
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

# ╔═╡ e84220a8-90bb-11eb-2fb8-adb6c87c2faa
const ROOT = "/home/nolln/root/data/seqspace/raw"

# ╔═╡ ce78d71a-917a-11eb-3cdd-b15aad75d147
const SAMPLE = 4

# ╔═╡ 0bb97860-917f-11eb-3dd7-cfd0c7d890cd
begin
	cdfplot(x; kwargs...) = plot(sort(x), range(0,1,length=length(x)); kwargs...)
	cdfplot!(x; kwargs...) = plot!(sort(x), range(0,1,length=length(x)); kwargs...)
end

# ╔═╡ be981c3a-90bb-11eb-3216-6bed955446f5
scRNA = ingredients("../src/scrna.jl").scRNA

# ╔═╡ f5ef8128-90bb-11eb-1f4b-053ed41f5038
begin 
	seq = scRNA.process(scRNA.load("$ROOT/rep$SAMPLE"));
	seq = scRNA.filtergene(seq) do gene, _
		sum(gene) >= 1e-2*length(gene) && maximum(gene) > 1
	end
end;

# ╔═╡ 9fdf7a60-91c5-11eb-3de6-471a838376d4
Inference = ingredients("../src/infer.jl").Inference

# ╔═╡ ca2dba38-9184-11eb-1793-f588473daad1
#seq = scRNA.generate(5000,1000);

# ╔═╡ 4f65301e-9186-11eb-1faa-71977e8fb097
let p
	p = cdfplot(
		vec(mean(seq.data,dims=2)),
		xscale=:log10,
		legend=false,
		linewidth=3
	)
	xaxis!("mean expression/cell")
	yaxis!("CDF")
	p
end

# ╔═╡ 005bd1ce-9185-11eb-2b21-d5cbec6adaa4
let p
	cm = ColorSchemes.inferno
	χ  = log.(vec(mean(null.data,dims=2)))
	Z  = maximum(χ)
	
	p = cdfplot(vec(null.cdf[1,:]),
		alpha=0.05,
		color=get(cm, χ[1]/Z), 
		label=""
	)
	for i ∈ sample(2:size(null.cdf,1), 2000; replace=false)
		cdfplot!(vec(null.cdf[i,:]),
			alpha=0.05,
			color=color=get(cm, χ[i]), 
			label=""
		)
	end
	
	xaxis!("theoretical cumalant")
	yaxis!("empirical cumalant")
	title!("simulated")
	
	p
end

# ╔═╡ 92dd32aa-91c5-11eb-2ddb-81162244af44
N, Z = Inference.sinkhorn_full(float.(seq.data)); N = N*size(seq,2); Z = Z/size(seq,2);

# ╔═╡ 0b84e778-91c6-11eb-295e-017e9b891fd2
r = nnmf(N, 300, verbose=true)

# ╔═╡ 2cac3764-91c6-11eb-39fc-3519e4ef5082
Ñ = r.W*r.H

# ╔═╡ a727a460-91c6-11eb-2211-33a36d4bf469
cor(Ñ[:],N[:])

# ╔═╡ 5197a480-91c8-11eb-0e53-716feab57083
size(Z)

# ╔═╡ 4624c51c-91c6-11eb-1764-0718f223e90c
scatter((Z.*Ñ)[2,:], seq[2,:])

# ╔═╡ 9594926e-91d0-11eb-22de-bdfe3290b19b
function resample(data)
	new = zeros(eltype(data), size(data))
	for g ∈ 1:size(data,1)
		new[g,:] = sample(vec(data[g,:]), size(new,2))
	end
	return new
end

# ╔═╡ aada5fb0-91c3-11eb-2338-91e3968f7769
F = svd(N);

# ╔═╡ cad5ecb4-91d0-11eb-0fcd-df07d605f091
F̂ = svd(resample(N));

# ╔═╡ 98a7f47a-91d1-11eb-20e7-5d8c55c28141
import PyPlot

# ╔═╡ 08a0ec2e-91d1-11eb-3d7b-1d52eef0d4f8
PyPlot.clf(); PyPlot.matshow(Z); PyPlot.clim(0,1e2); PyPlot.colorbar(); PyPlot.gcf()

# ╔═╡ 32698128-91d2-11eb-097c-673f5ba12ec8
II = 3; plot(F.U[:,II]); plot!(F̂.U[:,II], alpha=0.1)

# ╔═╡ b2c2a048-91c3-11eb-1ea8-b59ec3068cc4
plot(F.S/sum(F.S),yscale=:log10,xscale=:log10); plot!(F̂.S/sum(F̂.S),yscale=:log10)

# ╔═╡ c50bbc00-91c3-11eb-1476-bdf3cc711ccd
d=50; s̃ = F.U[:,1:d]*Diagonal(F.S[1:d])*F.Vt[1:d,:]

# ╔═╡ df97e704-91c3-11eb-0c82-bda474df78e2
scatter((Z.*s̃)[2,:], seq[2,:])

# ╔═╡ 0ac3487a-91d2-11eb-3443-393a61806665
corspearman((Z.*s̃)[:], seq[:])

# ╔═╡ 46d9f150-9188-11eb-2a31-05c636ece036
#Z, fits = scRNA.normalize(seq.data; algo=:discrete);

# ╔═╡ 32f72b3e-9189-11eb-224b-652cfa96ef48
#=let p
	cm = ColorSchemes.inferno
	χ  = log.(fits.χ)
	Z  = maximum(χ)
	
	p = cdfplot(fits.cdf[1],
		alpha=0.05,
		color=get(cm, χ[1]/Z),
		label=""
	)
	for i ∈ sample(2:length(fits.cdf), 2000; replace=false)
		p = cdfplot!(fits.cdf[i],
			alpha=0.05,
			color=get(cm, χ[i]/Z),
			label=""
		)
	end
	
	xaxis!("theoretical cdf")
	yaxis!("empirical cdf")
	title!("fits")
	p
end=#

# ╔═╡ 6f544b52-9189-11eb-3441-d56a79284d8d
#=let p
	p = scatter(seq.α, fits.α, 
		alpha=0.4, 
		marker_z=log10.(fits.χ),
		legend=false
	)
	xaxis!("true α")
	yaxis!("estimated α")
	p
end=#

# ╔═╡ 831b028e-9189-11eb-0e67-73495bf432fb
#=let p
	p = scatter(seq.β, fits.β, 
		alpha=0.4,
		marker_z=log10.(fits.χ),
		legend=false
	)
	xaxis!("true β")
	yaxis!("estimated β")
	p
end=#

# ╔═╡ bb739f56-9189-11eb-0b6e-bd72dd37a45f
#=let p
	p = scatter(seq.γ, 
		fits.γ, 
		alpha=0.4,
		marker_z=log10.(fits.χ),
		yscale=:log10, 
		xscale=:log10, 
		legend=false
	)
	xaxis!("true γ")
	yaxis!("estimated γ",(1e-1,1e5))
	p
end=#

# ╔═╡ 9251bd6c-919b-11eb-31f5-c921186904e9
#=let p
	p = scatter(fits.χ, 
		fits.γ,
		marker_z=log10.(fits.M),
		alpha=0.4,
		yscale=:log10, 
		xscale=:log10, 
		legend=false
	)
	xaxis!("mean expression/cell")
	yaxis!("estimated γ")
end=#

# ╔═╡ edfd6656-918d-11eb-3fbc-0d2d50765d89
#s = seq.data; s₁, s₂ = scRNA.bisect(s);

# ╔═╡ 27bec0d0-918e-11eb-2f82-73cde54d9893
#cor(vec(s₁[:]), vec(s₂[:]))

# ╔═╡ ceb63308-918e-11eb-2be0-ed6fadea7600
#=c = [ let
		ḡ = float.(vec(sum(s₁.data, dims=1)))
		r = nnmf(float.(s₁.data) * Diagonal(1 ./ ḡ), 
			k, 
			init=:nndsvdar, 
			alg=:multmse, 
			maxiter=250, 
			verbose=true
		)
		s̃ = (r.W*r.H) * Diagonal(ḡ)
		cor(log.(vec(s̃[:].+1e-3)), log.(vec(s₂[:].+1e-3)))
end for k ∈ [2, 5, 25, 50, 75, 100, 200, 300] ]=#

# ╔═╡ 2e41f6c6-9190-11eb-1016-430290dd8b67
#=let
	p = plot([2, 5, 25, 50, 75, 100, 200, 500], c)
	
	xaxis!("number of components")
	yaxis!("pearson correlation with complement")
	
	p
end=#

# ╔═╡ 1741edf0-919c-11eb-3501-bb5e40be67fc
#Z2, fits2 = scRNA.normalize(s; algo=:continuous, opt=:greedycd, k=50, δβ¯²=1e0);

# ╔═╡ c92e2f9e-919d-11eb-0c9f-33f59da11a29
#=let p
	cm = ColorSchemes.inferno
	χ  = log.(fits.χ)
	Z  = maximum(χ)
	
	p = cdfplot(fits2.cdf[1],
		alpha=0.05,
		color=get(cm, χ[1]/Z),
		label=""
	)
	for i ∈ sample(2:length(fits2.cdf), 2000; replace=false)
		p = cdfplot!(fits2.cdf[i],
			alpha=0.05,
			color=get(cm, χ[i]/Z),
			label=""
		)
	end
	
	xaxis!("theoretical cdf")
	yaxis!("empirical cdf")
	title!("fits")
	p
end=#

# ╔═╡ 7b64a0b6-919d-11eb-3d54-9113d80cd1d7
#=let p
	p = scatter(seq.α, fits2.α, 
		alpha=0.4, 
		marker_z=log10.(fits2.χ),
		legend=false
	)
	xaxis!("true α")
	yaxis!("estimated α")
	p
end=#

# ╔═╡ 8db73190-919d-11eb-1dc5-31b9ef5b6db3
#=let p
	p = scatter(seq.β, fits2.β, 
		alpha=0.4,
		marker_z=log10.(fits2.χ),
		legend=false
	)
	xaxis!("true β")
	yaxis!("estimated β")
	p
end=#

# ╔═╡ a415ff16-919d-11eb-0a12-534402c02a08
#=let p
	p = scatter(seq.γ, 
		fits2.γ, 
		alpha=0.4,
		marker_z=log10.(fits2.χ),
		yscale=:log10, 
		xscale=:log10, 
		legend=false
	)
	xaxis!("true γ")
	yaxis!("estimated γ")
	p
end=#

# ╔═╡ 86aed80a-91a0-11eb-04b4-8bcf20b9badc
#s̄ = s ./ sum(s,dims=1);

# ╔═╡ cd74215a-91a0-11eb-0663-f1fd821d0062
#S = log.(vec(mean(s,dims=1))); S = S / mean(S)

# ╔═╡ e6816d60-91af-11eb-0886-b131ab8317de
#c = [corspearman(vec(Z2[i,:]), vec(Z[i,:])) for i in 1:size(Z,1)]

# ╔═╡ f5bbc96a-91af-11eb-1c94-a509691fd535
#scatter(mean(s,dims=2), c, alpha=0.1)

# ╔═╡ 9749a6d6-919e-11eb-3cfe-9b1bf258f3c0
#=let i=100
	p = scatter(Z2[i,:], Z[i,:], 
		alpha=0.5, 
		marker_z=s[i,:], 
		markersize=5*S,
		label=""
	)
	
	xaxis!("gamma normalized")
	yaxis!("negative binomial normalized")
	p
end=#

# ╔═╡ 1f929770-91a9-11eb-3a82-038cea0bf31a
#F = svd(Z2)

# ╔═╡ 27934ac6-91a9-11eb-07a4-abda0d5a6851
#plot(F.S, xscale=:log10)

# ╔═╡ Cell order:
# ╟─2466c2a4-90c7-11eb-3f9c-5b87a7a35bb6
# ╟─969b3f50-90bb-11eb-2b67-c784d20c0eb2
# ╟─e84220a8-90bb-11eb-2fb8-adb6c87c2faa
# ╟─ce78d71a-917a-11eb-3cdd-b15aad75d147
# ╟─0bb97860-917f-11eb-3dd7-cfd0c7d890cd
# ╠═f5ef8128-90bb-11eb-1f4b-053ed41f5038
# ╠═be981c3a-90bb-11eb-3216-6bed955446f5
# ╠═9fdf7a60-91c5-11eb-3de6-471a838376d4
# ╠═ca2dba38-9184-11eb-1793-f588473daad1
# ╟─4f65301e-9186-11eb-1faa-71977e8fb097
# ╟─005bd1ce-9185-11eb-2b21-d5cbec6adaa4
# ╠═92dd32aa-91c5-11eb-2ddb-81162244af44
# ╠═0b84e778-91c6-11eb-295e-017e9b891fd2
# ╠═2cac3764-91c6-11eb-39fc-3519e4ef5082
# ╠═a727a460-91c6-11eb-2211-33a36d4bf469
# ╠═5197a480-91c8-11eb-0e53-716feab57083
# ╠═4624c51c-91c6-11eb-1764-0718f223e90c
# ╠═9594926e-91d0-11eb-22de-bdfe3290b19b
# ╠═aada5fb0-91c3-11eb-2338-91e3968f7769
# ╠═cad5ecb4-91d0-11eb-0fcd-df07d605f091
# ╠═98a7f47a-91d1-11eb-20e7-5d8c55c28141
# ╠═08a0ec2e-91d1-11eb-3d7b-1d52eef0d4f8
# ╠═32698128-91d2-11eb-097c-673f5ba12ec8
# ╠═b2c2a048-91c3-11eb-1ea8-b59ec3068cc4
# ╠═c50bbc00-91c3-11eb-1476-bdf3cc711ccd
# ╠═df97e704-91c3-11eb-0c82-bda474df78e2
# ╠═0ac3487a-91d2-11eb-3443-393a61806665
# ╠═46d9f150-9188-11eb-2a31-05c636ece036
# ╟─32f72b3e-9189-11eb-224b-652cfa96ef48
# ╟─6f544b52-9189-11eb-3441-d56a79284d8d
# ╟─831b028e-9189-11eb-0e67-73495bf432fb
# ╠═bb739f56-9189-11eb-0b6e-bd72dd37a45f
# ╠═9251bd6c-919b-11eb-31f5-c921186904e9
# ╠═edfd6656-918d-11eb-3fbc-0d2d50765d89
# ╠═27bec0d0-918e-11eb-2f82-73cde54d9893
# ╟─ceb63308-918e-11eb-2be0-ed6fadea7600
# ╟─2e41f6c6-9190-11eb-1016-430290dd8b67
# ╠═1741edf0-919c-11eb-3501-bb5e40be67fc
# ╠═c92e2f9e-919d-11eb-0c9f-33f59da11a29
# ╟─7b64a0b6-919d-11eb-3d54-9113d80cd1d7
# ╠═8db73190-919d-11eb-1dc5-31b9ef5b6db3
# ╠═a415ff16-919d-11eb-0a12-534402c02a08
# ╠═86aed80a-91a0-11eb-04b4-8bcf20b9badc
# ╠═cd74215a-91a0-11eb-0663-f1fd821d0062
# ╠═e6816d60-91af-11eb-0886-b131ab8317de
# ╠═f5bbc96a-91af-11eb-1c94-a509691fd535
# ╠═9749a6d6-919e-11eb-3cfe-9b1bf258f3c0
# ╠═1f929770-91a9-11eb-3a82-038cea0bf31a
# ╠═27934ac6-91a9-11eb-07a4-abda0d5a6851
