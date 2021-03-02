### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ a4a14288-7b90-11eb-01db-9dba870c417e
using WGLMakie, JSServe, PlutoUI

# ╔═╡ 56695996-7b8d-11eb-1612-07fab5dafac3
import Plots

# ╔═╡ c408a1fe-7b90-11eb-391f-23c775403ddf
Page()

# ╔═╡ c81b443e-7b90-11eb-1500-372fb43712b3
function makie(f::Function)
    scene = Scene(resolution = (600, 400))
	f(scene)
   
    return scene
end

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
	N  = 1000, 
	Ws = [50,50], 
	V  = 81,
	δ  = 1,
	γ  = 0,
)

# ╔═╡ 4ce81e9a-7b8b-11eb-2335-c960ad189017
result, data = M.SeqSpace.run(param); r = result[1]

# ╔═╡ 54f7b29c-7b8d-11eb-3fe4-f5d4ee676431
Plots.plot(r.loss.train, label="train"); Plots.plot!(r.loss.valid, label="validate")

# ╔═╡ d8066234-7b90-11eb-331e-9549202425c1
begin
	z = r.model.pullback(data.x)
	x̂ = data.map(r.model.pushforward(z))
end;

# ╔═╡ 085eb4dc-7b97-11eb-02c1-1761f9ed4e8c
begin
	scrna = data.map(data.x)
	invert, embryo = I.Inference.inversion(scrna, data.genes)
end

# ╔═╡ dbe1bdac-7b9c-11eb-0e3a-577e252a1f78
begin
	Ψ = invert(0.5)
	
	ψᵣ = Ψ  ./ sum(Ψ, dims=2)
	ψₗ = (Ψ ./ sum(Ψ, dims=1))'
end

# ╔═╡ 9b6544b0-7b9c-11eb-13aa-bf2ce2abeb55
r̂ = (ψₗ * embryo)'

# ╔═╡ fdedda4a-7b90-11eb-30ba-d9c76a5ccd08
makie() do s
	scatter!(s, z[1,:], z[2,:],color=r̂[2,:])
end

# ╔═╡ Cell order:
# ╠═56695996-7b8d-11eb-1612-07fab5dafac3
# ╠═a4a14288-7b90-11eb-01db-9dba870c417e
# ╠═c408a1fe-7b90-11eb-391f-23c775403ddf
# ╠═c81b443e-7b90-11eb-1500-372fb43712b3
# ╠═f90a2afc-7b8a-11eb-31ed-f79ca667aed0
# ╠═1abf567c-7b8b-11eb-1cd5-176f3985cf5f
# ╠═af24807a-7b96-11eb-031b-a554e04fd797
# ╠═36f6f034-7b8b-11eb-0f7c-3bf426e8c858
# ╠═4ce81e9a-7b8b-11eb-2335-c960ad189017
# ╠═54f7b29c-7b8d-11eb-3fe4-f5d4ee676431
# ╠═d8066234-7b90-11eb-331e-9549202425c1
# ╠═085eb4dc-7b97-11eb-02c1-1761f9ed4e8c
# ╠═dbe1bdac-7b9c-11eb-0e3a-577e252a1f78
# ╠═9b6544b0-7b9c-11eb-13aa-bf2ce2abeb55
# ╠═fdedda4a-7b90-11eb-30ba-d9c76a5ccd08
