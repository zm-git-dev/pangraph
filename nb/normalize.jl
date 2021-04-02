### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 2466c2a4-90c7-11eb-3f9c-5b87a7a35bb6
begin
	using LinearAlgebra, SpecialFunctions
	using Distributions, Statistics, StatsBase, Random
	using Optim, NLSolversBase
	using Clustering, Interpolations
	using Plots, ColorSchemes
	
	import GSL
	
	default(fmt = :png)
end

# ╔═╡ 7cf6be2e-9315-11eb-1cb1-396f2131908b
begin
	using JSServe
	import WGLMakie
end

# ╔═╡ b992e41a-9334-11eb-1919-87967d572a21
using JLD2, FileIO

# ╔═╡ fc2b03f0-924b-11eb-0f20-45edefca4b76
md"""
# Normalization

This notebook serves as a collection of thoughts on how to preprocess scRNAseq data
"""

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

# ╔═╡ be981c3a-90bb-11eb-3216-6bed955446f5
scRNA = ingredients("../src/scrna.jl").scRNA

# ╔═╡ e84220a8-90bb-11eb-2fb8-adb6c87c2faa
const ROOT = "/home/nolln/root/data/seqspace/raw"

# ╔═╡ ce78d71a-917a-11eb-3cdd-b15aad75d147
const SAMPLE = 4

# ╔═╡ 0bb97860-917f-11eb-3dd7-cfd0c7d890cd
begin
	cdfplot(x; kwargs...) = plot(sort(x), range(0,1,length=length(x)); kwargs...)
	cdfplot!(x; kwargs...) = plot!(sort(x), range(0,1,length=length(x)); kwargs...)
end

# ╔═╡ f972b674-9264-11eb-0a1c-0774ce2527e7
const ∞ = Inf

# ╔═╡ 9594926e-91d0-11eb-22de-bdfe3290b19b
function resample(data)
	new = zeros(eltype(data), size(data))
	for g ∈ 1:size(data,1)
		new[g,:] = sample(vec(data[g,:]), size(new,2))
	end
	return new
end

# ╔═╡ d7e28c3e-9246-11eb-0fd3-af6f94ea8562
function generate(ngene, ncell; ρ=(α=Gamma(0.25,2), β=Normal(1,.01)))
    N = zeros(Int, ngene, ncell)
	
    z = log.(rand(Gamma(5,1), ncell))
    α = log.(rand(ρ.α, ngene))
    β = rand(ρ.β, ngene)
	
    for g ∈ 1:ngene
        μ⃗ = exp.(α[g] .+ β[g].*z)
        for (c, μ) ∈ enumerate(μ⃗)
            N[g,c] = rand(Poisson(μ),1)[1]
        end
	end

    ι = vec(sum(N,dims=2)) .> 0

    N = N[ι, :]
    α = α[ι]
    β = β[ι]
		
	return (
		data  = N,
		param = (
			α = α,
			β = β,
			z = z,
		)
	)
end

# ╔═╡ ca2dba38-9184-11eb-1793-f588473daad1
null, params = generate(5000,2000);

# ╔═╡ bf8b0edc-9247-11eb-0ed5-7b1d16e00fc4
md"""
## How to find rank of count matrix?
### Prototype double stochastic
Assume we have a count matrix $X_{iα}$, where $i$ indexes genes and $\alpha$ indexes cells, that can be expressed as a matrix $Z_{i\alpha}$ of low rank $r$ plus a noise matrix of full rank $\mathcal{E}_{i\alpha}$.

``
X_{i\alpha} = Z_{i\alpha} + \mathcal{E}_{i\alpha}
``

By definition, the noise is expected to have zero mean

``
\langle \mathcal{E}_{i\alpha} \rangle = 0 \implies \langle X_{i\alpha} \rangle = Z_{i\alpha}
``

However, in general, we expect the noise matrix to be heteroskedastic, i.e. the variance of a given element of $\mathcal{E}$ will strongly depend upon its position. Thus the usual assumptions of a Marchenko-Pastur law are violated. 

For example, assume genes are Poisson distributed, but with different means depending on the exact gene. The variances of rows will then depend on this hidden variable. The same argument applies for considering columns. Thus, we look for a rescaling $\tilde{\mathcal{E}}$ that satisfies

``
G = \sum\limits_{i=1}^G \langle\tilde{\mathcal{E}}_{i\alpha}^2 \rangle \  \forall \alpha \qquad \text{and} \qquad N = \sum\limits_{\alpha=1}^N \langle\tilde{\mathcal{E}}_{i\alpha}^2 \rangle  \  \forall i
``

We define the rescaling by scaling rows by $\vec{u}$ and columns by $\vec{v}$, i.e.

``
\tilde{X}_{i\alpha} \equiv u_i X_{i\alpha} v_\alpha \implies \tilde{Z}_{i\alpha} \equiv u_i Z_{i\alpha} v_\alpha \quad \text{and} \quad \tilde{\mathcal{E}}_{i\alpha} \equiv u_i \mathcal{E}_{i\alpha} v_\alpha
``

#### Poisson example
Utilizing the Poisson example, we know the variance is equal to the mean and thus 

``
\langle\mathcal{E}_{i\alpha}^2 \rangle = X_{i\alpha} \implies \langle\tilde{\mathcal{E}}_{i\alpha}^2 \rangle = u^2_{i} X_{i\alpha} v^2_{\alpha}
``

Thus we can write down

``
G = \sum\limits_{i=1}^G  u^2_{i} X_{i\alpha} v^2_{\alpha} \  \forall \alpha \qquad \text{and} \qquad N = \sum\limits_{\alpha=1}^N u^2_{i} X_{i\alpha} v^2_{\alpha}  \  \forall i
``

We can solve for $\vec{u}$ and $\vec{v}$ by employing the Sinkhorn Knopp algorithm. We define the rescaled noise matrix by

``
\tilde{\Sigma}_{\alpha\beta} \equiv G^{-1} \sum\limits_{i} \tilde{\mathcal{E}}_{i\alpha}\tilde{\mathcal{E}}_{i\beta}
``

The distribution of eigenvalues of $\Sigma$ converge to Marchenko-Pastur with parameter of $\gamma = N/G$ and variance $1$, see *Girko, V. Theory of stochastic canonical equations 2001*. The largest eigenvalue is expected to be 

``\lambda_{max} \approx (1+\sqrt{\gamma})^2``

Thus the rank is given by the number of eigenvectors of $\Sigma$ greater than $\lambda_{max}$.
"""

# ╔═╡ f387130c-924f-11eb-2ada-794dfbf4d30a
function sinkhorn(A; r=[], c=[], maxᵢₜ=1000, δ=1e-4, verbose=false)
	if length(r) == 0
		r = size(A,2)
	end
	
	if length(c) == 0
		c = size(A,1)
	end
	
	x = ones(size(A,1))
	y = ones(size(A,2))
	for i ∈ 1:maxᵢₜ
		δr = maximum(abs.(x.*(A*y)  .- r))
		δc = maximum(abs.(y.*(A'*x) .- c))
		
		if verbose
			@show minimum(A'*x), minimum(A*y)
			@show r, c
			@show i, δr, δc
		end
		
		(isnan(δr) || isnan(δc)) && return x, y, false
		(δr < δ && δc < δ) 		 && return x, y, true
		
		y = c ./ (A'*x)
		x = r ./ (A*y)
		if verbose
			@show mean(x), mean(y)
		end
	end
	
	return x, y, false
end

# ╔═╡ eed136bc-924d-11eb-3e3a-374d21772e4b
u², v² = sinkhorn(null; verbose=true)

# ╔═╡ 7f38ceba-9253-11eb-000f-25045179e841
X̃ = (Diagonal(.√u²) * null * Diagonal(.√v²)); Σ̃ = X̃'*X̃ / size(X̃,1);

# ╔═╡ b14e4f0e-9253-11eb-171a-053dcc942240
let
	λ = eigvals(Σ̃)
	p = cdfplot(λ, xscale=:log10, linewidth=2, label="Empirical distribution")
	
	vline!([(1+sqrt(size(X̃,2)/size(X̃,1)))^2], linewidth=2, linestyle=:dashdot, label="MP maximum eigenvalue")
	title!("poisson noise")
	xaxis!("eigenvalue")
	yaxis!("CDF")
	p
end

# ╔═╡ cbcbb038-9255-11eb-0fc3-3ba4d95cee62
function generate_mean(null, rank)
	A = exp.(rand(Uniform(-1,2), (size(null,1),rank)))
	B = rand(Uniform(0,1), (rank,size(null,2)))

	return first.(rand.(Poisson.(A*B),1))
end

# ╔═╡ b8bb97aa-9256-11eb-253f-a16885888c5f
let
	X = generate_mean(null, 100) + null
	u², v² = sinkhorn(X; verbose=true)
	X̃ = (Diagonal(.√u²) * X * Diagonal(.√v²))
	Σ̃ = X̃'*X̃ / size(X̃,1);
	
	λ = eigvals(Σ̃)
	p = cdfplot(λ, xscale=:log10, linewidth=2, label="Empirical distribution")
	
	vline!([(1+sqrt(size(X̃,2)/size(X̃,1)))^2], linewidth=2, linestyle=:dashdot, label="MP maximum eigenvalue")
	title!("rank 100")
	xaxis!("eigenvalue")
	yaxis!("CDF")
	
	p
end

# ╔═╡ 5b25177c-925d-11eb-1aec-f52c8c52ec93
md"""
#### Beyond Poisson
At first glance this looks great. Unfortunately, we don't expect our count data to be Poisson distributed - scRNAseq data is generically overdispersed. Thus the relation that allowed us to equate the variance to the observed count matrix doesn't hold. 

For example, consider a negative binomial distribution (NB1) where 
``
\sigma^2 = \mu(1+\gamma)
``
In our previous notation,
``
\langle \mathcal{E}_{i\alpha}^2 \rangle = (1+\gamma_i) X_{i\alpha}
``
, where we have allowed for a gene-dependent overdispersion parameter. Working through the rescaling
``
\langle \tilde{\mathcal{E}}_{i\alpha}^2 \rangle = (1+\gamma_i) u^2_i X_{i\alpha} v^2_\alpha 
``
such that

``
G = \sum\limits_{i=1}^G  u^2_{i} (1+\gamma_i) X_{i\alpha} v^2_{\alpha} \  \forall \alpha \qquad \text{and} \qquad N = \sum\limits_{\alpha=1}^N u^2_{i} (1+\gamma_i) X_{i\alpha} v^2_{\alpha}  \  \forall i
``

This would require us to estimate the gene-specific overdispersion before performing further analyses. Similarly, for the NB2 parameterization, where
``
\sigma^2 = \mu(1+\gamma^{-1}\mu)
``, 
such that
``
\langle \mathcal{E}_{i\alpha}^2 \rangle = (1+\gamma_i^{-1}\langle X_{i\alpha} \rangle) \langle X_{i\alpha} \rangle
``
. Unbiased estimates of the variance give us

``
\langle \mathcal{E}_{i\alpha}^2 \rangle = \frac{X_{i\alpha}(X_{i\alpha} + \gamma_i)}{1+\gamma_i} \implies \langle \tilde{\mathcal{E}}_{i\alpha}^2 \rangle = u^2_{i} \frac{X_{i\alpha}(X_{i\alpha} + \gamma_i)}{1+\gamma_i} v^2_{\alpha}
``

In general, given an unbiased estimation for the variance we simply apply Sinkhorn-Knopp to obtain

``
G = \sum\limits_{i=1}^G  u^2_{i} \,\text{Var}[X]_{i\alpha} \, v^2_{\alpha} \  \forall \alpha \qquad \text{and} \qquad N = \sum\limits_{\alpha=1}^N u^2_{i} \, \text{Var}[X]_{i\alpha} \, v^2_{\alpha}  \  \forall i
``

A priori it's unclear if we could reliably detect $\gamma_i$ at the low counts we observe in-vivo

#### GLM formulation
scRNAseq data will have systematic variations in cell-depth simply as a technological consequence. Thus we must fit our distributions sensitive to this fact. The simple solution would be to divide all cells by their total sequencing depth. We opt for a similar, yet more flexible approach, namely we allow for the mean of the negative binomial to be dependent on the sequencing depth for each cell, $z_\alpha$. Specifically, we take the mean of the negative binomial distribution for gene $i$ and cell $\alpha$ to be

``
\text{log}(\mu_{i\alpha}) = \alpha_i + \beta_i * \text{log}(z_\alpha)
``

``\alpha_i`` and ``\beta_i``, along with the overdispersion parameter $\gamma_i$ define 3 parameters per gene we fit from raw count data using a Maximum Likelihood formalism
"""

# ╔═╡ c65e8a86-9259-11eb-29bb-3bf5c089746f
function generate_nb2(ngene, ncell; ρ=(α=Gamma(0.25,2), β=Normal(1,.01), γ=Gamma(3,3)))
    N = zeros(Int, ngene, ncell)
	
    z = log.(rand(Gamma(5,1), ncell))
    α = log.(rand(ρ.α, ngene))
    β = rand(ρ.β, ngene)
	γ = rand(ρ.γ, ngene)
	
    for g ∈ 1:ngene
        μ⃗ = exp.(α[g] .+ β[g].*z)
        for (c, μ) ∈ enumerate(μ⃗)
			λ = rand(Gamma(γ[g], μ/γ[g]),1)[1]
            N[g,c] = rand(Poisson(λ),1)[1]
        end
	end

	@show maximum(N, dims=2)
    ι = (vec(sum(N,dims=2)) .> 0) .& (vec(maximum(N,dims=2)) .> 1)

    N = N[ι, :]
    α = α[ι]
    β = β[ι]
	γ = γ[ι]
	
	return (
		data  = N,
		param = (
			α = α,
			β = β,
			γ = γ,
			z = z,
		),
	)
end

# ╔═╡ 5ebfd944-9262-11eb-3740-37ba8930e1c6
begin

function loss_nb1(x⃗, z⃗, β̄, δβ¯²)
	function f(Θ)
		α, β, γ = Θ
		
		Mu = (exp(+α + β*z) for z ∈ z⃗)
        S  = (loggamma(x+γ*μ) 
			- loggamma(x+1) 
			- loggamma(γ*μ) 
			+ x*log(γ) 
			- (x+γ*μ)*log(1+γ) for (x,μ) ∈ zip(x⃗,Mu))

        return -mean(S) + 0.5*δβ¯²*(β-β̄)^2
	end
	
	return f
end
	
function loss_nb2(x⃗, z⃗, β̄, δβ¯²)
	function f(Θ)
		α, β, γ = Θ
		
        S  = (loggamma(x+γ) 
			- loggamma(x+1) 
			- loggamma(γ) 
			+ x*(α+β*z)
			+ γ*log(γ)
			- (x+γ)*log(exp(α+β*z)+γ) for (x,z) ∈ zip(x⃗,z⃗))

        return -sum(S) + 0.5*δβ¯²*(β-β̄)^2
	end
	
	return f
end

function fit1(x, z, β̄, δβ¯²)
	μ  = mean(x)
	Θ₀ = [
		log(μ),
		β̄,
		var(x)/μ - 1,
	]
	
    if Θ₀[end] < 0 || isinf(Θ₀[end])
        Θ₀[end] = 1
	end
	
	loss = 	TwiceDifferentiable(
        loss_nb2(x, z, β̄, δβ¯²),
		Θ₀;
		autodiff=:forward
	)
	
	constraint = TwiceDifferentiableConstraints(
		[-∞, -∞, 0],
		[+∞, +∞, +∞],
	)

	soln = optimize(loss, constraint, Θ₀, IPNewton())
	
	Θ̂  = Optim.minimizer(soln)
	Ê  = Optim.minimum(soln)
	δΘ̂ = diag(inv(hessian!(loss, Θ̂)))
	
	# pearson residuals
	α̂, β̂, γ̂ = Θ̂
	μ̂ = exp.(α̂ .+ β̂*z)
	σ̂ = .√(μ̂ .+ μ̂.^2 ./ γ̂)
    ρ = (x .- μ̂) ./ σ̂

	return (
		parameters=Θ̂, 
		uncertainty=δΘ̂, 
		likelihood=Ê,
		trend=μ̂,
		residuals=ρ,
	)
end

function fit(data; β̄=1, δβ¯²=1e-2)
	z = log.(vec(mean(data, dims=1)))
	χ = log.(vec(mean(data, dims=2)))

    fits = [begin 
				@show i 
				fit1(vec(data[i,:]), z, β̄, δβ¯²) 
			end for i ∈ 1:size(data,1)]
	
	return vcat((fit.residuals' for fit ∈ fits)...),
        (
            likelihood  = map((f)->f.likelihood,  fits),

            α  = map((f)->f.parameters[1],  fits),
            β  = map((f)->f.parameters[2],  fits),
            γ  = map((f)->f.parameters[3],  fits),

            δα = map((f)->f.uncertainty[1], fits),
            δβ = map((f)->f.uncertainty[2], fits),
            δγ = map((f)->f.uncertainty[3], fits),

            μ̂ = map((f)->f.trend,  fits),
            χ = vec(mean(data, dims=2)),
            M = vec(maximum(data, dims=2))
        )
end
	
end

# ╔═╡ b563c86c-9264-11eb-01c2-bb42d74c3e69
E, p = generate_nb2(2000,1000);

# ╔═╡ 4f65301e-9186-11eb-1faa-71977e8fb097
let p
	p = cdfplot(
		vec(mean(null,dims=2)),
		xscale=:log10,
		legend=false,
		linewidth=3
	)
	xaxis!("mean expression/cell")
	yaxis!("CDF")
	p
end

# ╔═╡ c423eca6-9264-11eb-375f-953e02fc7ec4
Z, p̂ = fit(E; δβ¯²=10);

# ╔═╡ fb056230-9265-11eb-1a98-33c234a0f959
let
	p = scatter(p.α, p̂.α, alpha=0.1, marker_z=log10.(p̂.χ), label=false)
	xaxis!("true α")
	yaxis!("estimated α")
	p
end

# ╔═╡ c8c4e6b4-9266-11eb-05e6-917b82a580ab
let
	p = scatter(p.β, p̂.β, alpha=0.1, marker_z=log10.(p̂.χ), label=false)
	xaxis!("true β")
	yaxis!("estimated β")
	p
end

# ╔═╡ 622247ba-9268-11eb-3c0b-07f9cd3c6236
let
	p = scatter(p.γ, p̂.γ, xscale=:log10, yscale=:log10, marker_z=log10.(p̂.M), alpha=0.1, label=false)
	xaxis!("true γ")
	yaxis!("estimated γ")
	p
end

# ╔═╡ d9880e94-92f6-11eb-3f1e-cf4462c3b89a
let
	p = scatter(p̂.χ, p̂.γ, xscale=:log10, yscale=:log10, marker_z=log10.(p̂.M), alpha=0.1, label=false)
	xaxis!("expression per cell")
	yaxis!("estimated γ")
	vline!([1], label="cutoff", linestyle=:dashdot, linewidth=3)

	p
end

# ╔═╡ 17395b62-9272-11eb-0237-430f2e6499d6
let
	V = E.*(E.+p̂.γ) ./ (1 .+ p̂.γ)
	u², v² = sinkhorn(V; verbose=true)
	X̃ = (Diagonal(.√u²) * E * Diagonal(.√v²))
	Σ̃ = X̃'*X̃ / size(X̃,1);
	
	λ = eigvals(Σ̃)
	p = cdfplot(λ, xscale=:log10, linewidth=2, label="Empirical distribution")
	
	vline!([(1+sqrt(size(X̃,2)/size(X̃,1)))^2], linewidth=2, linestyle=:dashdot, label="MP maximum eigenvalue")
	title!("rank 0 (overdispersed)")
	xaxis!("eigenvalue")
	yaxis!("CDF")
	
	p
end

# ╔═╡ 3886d34c-9279-11eb-31e6-0fd49a2694aa
let
	E, p = generate_nb2(2000,1000);
	X 	 = generate_mean(E, 100) + E
	
	Z, p̂ = fit(X; δβ¯²=10);
	V 	 = X.*(X.+p̂.γ) ./ (1 .+ p̂.γ)

	u², v² = sinkhorn(V; verbose=true)

	X̃ = (Diagonal(.√u²) * X * Diagonal(.√v²))
	Ṽ = (Diagonal(u²) * V * Diagonal(v²))
	#Σ̃ = V'*V / size(X̃,1);
	
	λ = svdvals(X̃) #eigvals(Σ̃)
	p = cdfplot(λ, xscale=:log10, linewidth=2, label="Empirical distribution")
	k = sum(λ .>= (sqrt(size(X̃,1))+sqrt(size(X̃,2))-12))
	
	vline!([(sqrt(size(X̃,1))+sqrt(size(X̃,2))-12)], linewidth=2, linestyle=:dashdot, label="MP maximum eigenvalue")
	title!("rank ≈ $k (overdispersed)")
	xaxis!("singular value")
	yaxis!("CDF")
	
	p
end

# ╔═╡ 3d3a37d6-927d-11eb-16b3-e74d10851013
X = generate_mean(E, 100) + E;

# ╔═╡ 8472a6a6-927d-11eb-0992-897698a13544
Z₂, p̂₂ = fit(X; δβ¯²=10);

# ╔═╡ a82fe0da-926a-11eb-063b-f70bd587a789
R = kmeans(log.(hcat(p̂.χ, p̂.γ)'), 2);

# ╔═╡ a79aaeb0-926b-11eb-32ea-2bc1ead29909
function trendline(x, y; n = 10)
	l,r = log(minimum(x)), log(maximum(x))
    bp  = range(l, r, length=n+1)
	
    x₀ = Array{eltype(x),1}(undef, n)
    μ  = Array{eltype(y),1}(undef, n)
    for i ∈ 1:n
        xₗ, xᵣ = exp(bp[i]), exp(bp[i+1])
        pts = y[xₗ .≤ x .≤ xᵣ]
        if length(pts) > 0
            μ[i] = exp.(mean(log.(pts)))
        else
            μ[i] = μ[i-1]
        end
        x₀[i] = 0.5*(xₗ + xᵣ)
    end

    x₀ = [exp(l); x₀; exp(r)]
    μ  = [y[argmin(x)]; μ; y[argmax(x)]]
    return extrapolate(
        interpolate((x₀,), μ, Gridded(Linear())), 
        Line()
    )
end

# ╔═╡ f53061cc-92f4-11eb-229b-b9b501c6cab8
md"""
## Real data

Let's try this procedure on a single run of scRNAseq on Drosophila. We first run our negative binomial fitting with no priors put on parameter estimates.
"""

# ╔═╡ f5ef8128-90bb-11eb-1f4b-053ed41f5038
begin 
	seq = scRNA.process(scRNA.load("$ROOT/rep$SAMPLE"));
	seq = scRNA.filtergene(seq) do gene, _
		sum(gene) >= 1e-2*length(gene) && length(unique(gene)) > 2
	end
end;

# ╔═╡ f6f99ff4-92f5-11eb-2577-e51ab1cacfa6
S, p̃ = fit(seq; δβ¯²=100);

# ╔═╡ 65224940-92f6-11eb-3045-4fc4b7b29a6c
let
	p = scatter(p̃.χ, p̃.α, 
		alpha=0.1, 
		xscale=:log10, 
		marker_z=log10.(p̃.M), 
		label=false
	)
	xaxis!("expression / cell")
	yaxis!("estimated α")
	p
end

# ╔═╡ 992dd34e-92f6-11eb-2642-bb9862148733
let
	p = scatter(p̃.χ, p̃.β, 
		alpha=0.1, 
		xscale=:log10, 
		marker_z=log10.(p̃.M), 
		label=false
	)
	xaxis!("expression / cell")
	yaxis!("estimated β")
	p
end

# ╔═╡ a453ad84-92f6-11eb-0350-d1fbfbbbfda0
let
	p = scatter(p̃.χ, p̃.γ, 
		alpha=0.1, 
		xscale=:log10,
		yscale=:log10,
		marker_z=log10.(p̃.M), 
		label=false
	)
	xaxis!("expression / cell")
	yaxis!("estimated γ")
	vline!([1], label="cutoff", linestyle=:dashdot, linewidth=3)
	p
end

# ╔═╡ f9d549ac-92f6-11eb-306a-77bf90c8eb33
md"""
We observe a pattern in fitting negative binomial distributions in both the synthetic data (where we know γ does not depend upon χ) and in the real data. Specifically, there exists a knee at around average expression of $1$ where γ becomes independent of expression. Below, there is a roughly linear decrease with a large spread.

Thus we make the ansatz that γ does not actually depend upon α (they are independent variables) but rather this is an artifact of sparse sampling. As such we find the distribution of γ from highly expressed genes and subsequently use our estimate as a prior in the inference of the distribution for the remaining genes to help constrain our inference.
"""

# ╔═╡ 94b6d52a-92f8-11eb-226f-27e28f4a4d4b
function fit_gamma(data)
	L = (Θ) -> let
		k, θ = Θ
		return -sum((k-1)*log.(data) .- (data./θ) .- loggamma(k) .- k*log(θ))
	end
	
	μ  = mean(data)
	σ² = var(data)
	Θ₀ = [
		μ^2/σ²,
		σ²/μ,
	]
	
	loss = TwiceDifferentiable(
        L,
		Θ₀;
		autodiff=:forward
	)
	
	constraint = TwiceDifferentiableConstraints(
		[0,  0],
		[+∞, +∞],
	)

	soln = optimize(loss, constraint, Θ₀, IPNewton())

	return Optim.minimizer(soln)
end

# ╔═╡ 4f96560a-92fd-11eb-19f7-7b11f0ee46bf
function fit_lognormal(data)
	L = (Θ) -> let
		μ, σ = Θ
		return sum(log(σ) .+ (log.(data) .- μ).^2 ./ (2*σ^2))
	end
	
	μ = mean(log.(data))
	σ = std(log.(data))
	Θ₀ = [
		μ,
		σ
	]
	
	loss = TwiceDifferentiable(
        L,
		Θ₀;
		autodiff=:forward
	)
	
	constraint = TwiceDifferentiableConstraints(
		[0,  0],
		[+∞, +∞],
	)

	soln = optimize(loss, constraint, Θ₀, IPNewton())

	return Optim.minimizer(soln)
end

# ╔═╡ 1f2bdf42-92ff-11eb-0e1f-e7c4e1e78058
function fit_gennormal(data)
	L = (Θ) -> let
		μ, σ, β = Θ
		return sum(loggamma(1/β) + log(σ) .+ (abs.(data .- μ)./ σ).^β .- log(β))
	end
	
	μ = mean(data)
	σ = std(data)
	Θ₀ = [
		μ,
		σ,
		2
	]
	
	loss = TwiceDifferentiable(
        L,
		Θ₀;
		autodiff=:forward
	)
	
	constraint = TwiceDifferentiableConstraints(
		[0,  0,  0],
		[+∞, +∞, +∞],
	)

	soln = optimize(loss, constraint, Θ₀, IPNewton())

	return Optim.minimizer(soln)
end

# ╔═╡ bc1990b6-92f7-11eb-0015-310aefc5041e
let
	rank(x) = invperm(sortperm(x))
	
 	Y = p̃.γ[p̃.χ .> 1.05]
	
	k̂, θ̂ = fit_gamma(Y)
	@show k̂, θ̂
	
	μ̂, σ̂ = fit_lognormal(Y)
	@show μ̂, σ̂
	
	ŷ, α̂, β̂ = fit_gennormal(log.(Y))
	@show ŷ, α̂, β̂

	GG(x)  = first(gamma_inc(k̂, (x/θ̂), 0))
	LN(x) = .5*(1+erf((log(x) .- μ̂) ./ (sqrt(2)*σ̂)))
	GN(x) = .5*(1+sign(log(x)-ŷ)*first(gamma_inc((1/β̂), abs(log(x)-ŷ)/(α̂^β̂), 0)))

	x = range(minimum(Y),maximum(Y),length=100)
	
	scatter(rank(Y)/length(Y),  GG.(Y), alpha=0.5, label="gamma distribution")
	scatter!(rank(Y)/length(Y), LN.(Y), alpha=0.5, label="log-normal distribution")
	scatter!(rank(Y)/length(Y), GN.(Y), alpha=0.5, label="log-generalized-normal distribution")

	plot!(0:1, 0:1, linewidth=2, linecolor=:black, linestyle=:dashdot, label="ideal", legend=:bottomright)
	
	xaxis!("Empirical CDF")
	yaxis!("Fit CDF")
end

# ╔═╡ 9f56e6fc-9303-11eb-2bdd-5da078b4d9c3
md"""
We try out three different distributions to see which best estimates the empirical distribution.
  * Gamma distribution appears too heavy right tailed.
  * Log-normal is a better approximation, however we see it is still too heavy tailed (lack of values in the bulk of the distribution implies we are chasing the tails)
  * Generalized log-normal (which tunes the contribution of the tails) appears adequate.
"""

# ╔═╡ 37afad28-9304-11eb-2f4d-8397fca7bb99
begin

function loss_nb2_constrained(x⃗, z⃗, β̄, δβ¯², σ̂, ν̂, μ̂)
	function f(Θ)
		α, β, γ = Θ
		
        S  = (loggamma(x+γ) 
			- loggamma(x+1) 
			- loggamma(γ) 
			+ x*(α+β*z)
			+ γ*log(γ)
			- (x+γ)*log(exp(α+β*z)+γ) for (x,z) ∈ zip(x⃗,z⃗))

        return -sum(S) + 0.5*δβ¯²*(β-β̄)^2 + (abs.(log(γ)-μ̂)/σ̂)^ν̂
	end
	
	return f
end

function fit1_constrained(x, z, β̄, δβ¯², σ̂, ν̂, μ̂)
	μ  = mean(x)
	Θ₀ = [
		log(μ),
		β̄,
		var(x)/μ - 1,
	]
	
    if Θ₀[end] < 0 || isinf(Θ₀[end])
        Θ₀[end] = 1
	end
	
	loss = 	TwiceDifferentiable(
        loss_nb2_constrained(x, z, β̄, δβ¯², σ̂, ν̂, μ̂),
		Θ₀;
		autodiff=:forward
	)
	
	constraint = TwiceDifferentiableConstraints(
		[-∞, -∞, 0],
		[+∞, +∞, +∞],
	)

	soln = optimize(loss, constraint, Θ₀, IPNewton())
	
	Θ̂  = Optim.minimizer(soln)
	Ê  = Optim.minimum(soln)
	δΘ̂ = diag(inv(hessian!(loss, Θ̂)))
	
	# pearson residuals
	α̂, β̂, γ̂ = Θ̂
	μ̂ = exp.(α̂ .+ β̂*z)
	σ̂ = .√(μ̂ .+ μ̂.^2 ./ γ̂)
    ρ = (x .- μ̂) ./ σ̂

	return (
		parameters=Θ̂, 
		uncertainty=δΘ̂, 
		likelihood=Ê,
		trend=μ̂,
		residuals=ρ,
	)
end

function fit_constrained(data; β̄=1, δβ¯²=1e-2, σ̂, ν̂, μ̂)
	z = log.(vec(mean(data, dims=1)))
	χ = log.(vec(mean(data, dims=2)))

    fits = [begin 
				@show i 
				fit1_constrained(vec(data[i,:]), z, β̄, δβ¯², σ̂, ν̂, μ̂) 
			end for i ∈ 1:size(data,1)]
	
	return vcat((fit.residuals' for fit ∈ fits)...),
        (
            likelihood  = map((f)->f.likelihood,  fits),

            α  = map((f)->f.parameters[1],  fits),
            β  = map((f)->f.parameters[2],  fits),
            γ  = map((f)->f.parameters[3],  fits),

            δα = map((f)->f.uncertainty[1], fits),
            δβ = map((f)->f.uncertainty[2], fits),
            δγ = map((f)->f.uncertainty[3], fits),

            μ̂ = map((f)->f.trend,  fits),
            χ = vec(mean(data, dims=2)),
            M = vec(maximum(data, dims=2))
        )
end
	
end

# ╔═╡ d2e5b674-9305-11eb-1254-5b9fa030e8ee
Sᵪ, p̃ᵪ = let
	Y = p̃.γ[p̃.χ .> 1.05]
	μ, σ, ν = fit_gennormal(log.(Y))
	fit_constrained(seq; δβ¯²=100, μ̂=μ, σ̂=σ, ν̂=ν);
end;

# ╔═╡ 70fc4538-9306-11eb-1509-3dfa1034d8aa
let
	p = scatter(p̃ᵪ.χ, p̃ᵪ.α, 
		alpha=0.1, 
		xscale=:log10, 
		marker_z=log10.(p̃ᵪ.δα), 
		label=false
	)
	xaxis!("expression / cell")
	yaxis!("estimated α")
	p
end

# ╔═╡ 861ca2b6-9306-11eb-311c-69c6984faa26
let
	p = scatter(p̃ᵪ.χ, p̃ᵪ.β, 
		alpha=0.1, 
		xscale=:log10, 
		marker_z=log10.(p̃ᵪ.δβ./p̃ᵪ.β), 
		label=false
	)
	xaxis!("expression / cell")
	yaxis!("estimated β")
	p
end

# ╔═╡ 97ded136-9306-11eb-019b-636b739a61d6
let
	p = scatter(p̃ᵪ.χ, p̃ᵪ.γ, 
		alpha=0.1, 
		xscale=:log10,
		yscale=:log10,
		marker_z=log10.(p̃ᵪ.δγ./p̃ᵪ.γ), 
		label=false
	)
	xaxis!("expression / cell")
	yaxis!("estimated γ")
	vline!([1], label="cutoff", linestyle=:dashdot, linewidth=3)
	p
end

# ╔═╡ e05d2a98-9306-11eb-252e-5baf6fc59f8f
let 
	V = seq.*(seq.+p̃ᵪ.γ) ./ (1 .+ p̃ᵪ.γ)
	u², v² = sinkhorn(V; verbose=true)

	X̃ = (Diagonal(.√u²) * seq * Diagonal(.√v²))
	Ṽ = (Diagonal(u²) * V * Diagonal(v²))
	
	λ = svdvals(X̃)
	p = cdfplot(λ, xscale=:log10, linewidth=2, label="Empirical distribution")
	
	ϵ = 0
	k = sum(λ .> (sqrt(size(X̃,1))+sqrt(size(X̃,2))) - ϵ)
	
	vline!([(sqrt(size(X̃,1))+sqrt(size(X̃,2))-ϵ)], linewidth=2, linestyle=:dashdot, label="MP maximum eigenvalue")
	title!("scRNAseq data (k=$k)")
	xaxis!("singular value")
	yaxis!("CDF")
end

# ╔═╡ bdffde54-9307-11eb-2ccb-ed48777f28f8
md"""
Remarkably this works beautifully! We recover similar (but smaller as expected) estimates for the true rank of the matrix. Furthermore, we have estimated a scaling of our matrix such that every row and column sum have unit variance. We treat this as our processed matrix going forward.

##### Idea:
One could imagine futher iterating on the estimate for the γ prior as we do see some expression dependence in the inferred values even though the prior does not. This would be tantamount to repeating the same procedure in bins and then interpolating the fit values.

But for now we consider this a success. We should throw out genes with excessive uncertainty in $γ$.
"""

# ╔═╡ 91742bde-9309-11eb-2acc-836cc1ab1aee
S̃, Ṽ = let 
	V = seq.*(seq.+p̃ᵪ.γ) ./ (1 .+ p̃ᵪ.γ)
	u², v² = sinkhorn(V; verbose=true)

	(Diagonal(.√u²) * seq * Diagonal(.√v²)), (Diagonal(u²) * V * Diagonal(v²))
end

# ╔═╡ df6e73de-9309-11eb-3bd3-3f9f511744cf
md"""
## Data imputation
Can we utilize the large number of genes to help "impute" the dropout? We tread lightly here, any averaging over cells will introduce non-trivial correlations in the data.

The basic idea is to compute a distance matrix between cells $D_{\alpha\beta}$. You can then use this to define a Gaussian kernel $K_{\alpha\beta} \sim e^{-D^2_{\alpha\beta}}$ where $K$ is assumed to be suitably normalized. Data is then "imputed" by averaging, i.e. a given count matrix $X_{i\alpha}$ is averaged $\tilde{X}_{i\alpha} = X_{i\alpha} K^t_{\alpha\beta}$ where $t$ is the diffusion time.

This assumes there is a low-dimensional manifold on which our data lives. The above considerations do imply this. Thus the thought experiment is as follows: if we smooth our data and then fit our distributions, can we improve our fits of the underlying distributions, i.e. deal with our sparse sampling problem?

Let's take our scaled matrix (of unit variance) and see what we can find. In all distance measures tested, we only utilize them within a small neighborhood - geodesic distances are used outside.
"""

# ╔═╡ 071f2c26-930c-11eb-2745-93cb2001e76b
PointCloud = ingredients("../src/geo.jl").PointCloud

# ╔═╡ c8227e0e-9326-11eb-2724-cb588170c7c2
Inference = ingredients("../src/infer.jl").Inference

# ╔═╡ 8ffaa54e-930b-11eb-2f1f-9907008b76d2
md"""
#### Euclidean
"""

# ╔═╡ dd9986f4-930f-11eb-33b8-67dbc4d1d087
S̃ₐ = let d = 35
	F = svd(S̃);
	F.U[:,1:d]*Diagonal(F.S[1:d])*F.Vt[1:d,:]
end;

# ╔═╡ 3c979aa0-930c-11eb-3e6e-9bdf7f3029b5
Gₑ = PointCloud.geodesics(S̃ₐ, 6); size(Gₑ)

# ╔═╡ ed6333c6-930c-11eb-25c0-c551592746e0
let
	ρ, Rs = PointCloud.scaling(Gₑ, 1000)

	p = plot(Rs, ρ', alpha=0.03, color=:cyan3, label="")
	plot!(Rs, mean(ρ, dims=1)', linewidth=5, alpha=1, color=:cyan3, label="mean")

	xaxis!("radius", :log10)
	yaxis!("number of points", :log10)

	plot!(Rs, 5e-4*Rs.^3, linestyle=:dashdot, linewidth=2, alpha=1, color=:coral3, label="αR³", legend=:bottomright)

	xaxis!("radius", :log10, (20, 400))
	yaxis!("number of points", :log10, (6, 1200))

	p
end

# ╔═╡ 76b9c454-9316-11eb-1548-bdef70d532d6
md"""
#### Aside:
Embedding estimate for points
"""

# ╔═╡ 585bed74-9319-11eb-2854-9d74e1d592c2
GENE = findfirst(seq.gene .== "sna")

# ╔═╡ 91147112-9315-11eb-0a71-8ddfdb2497a6
Page()

# ╔═╡ e96bc054-9310-11eb-25ca-576caeb336b8
Kₑ = let
	σ = .15*mean(Gₑ[:])
	K = exp.(-(Gₑ./σ).^2)
	u, v = sinkhorn(K; r=1, c=1)
	Diagonal(u) * K * Diagonal(v)
end;

# ╔═╡ f7d6acc2-9314-11eb-03f3-f9f221ded27c
let
	ξ = PointCloud.mds(Gₑ.^2, 3)
	WGLMakie.scatter(ξ[:,1], ξ[:,2], ξ[:,3], color=Kₑ*Sᵪ[GENE,:], markersize=4000)
end

# ╔═╡ 7a3b0202-9311-11eb-101e-99fbc40c6633
let
	S = seq.data ./ sum(seq.data,dims=1)
	Ñ = (S*Kₑ) .* sum(seq.data,dims=1)
	c = [ cor(vec(Ñ[i,:]), vec(seq[i,:])) for i in 1:size(Ñ,1) ]

	cdfplot(c, linewidth=2, label="")
	
	xaxis!("correlation before/after smoothing")
	yaxis!("CDF")
end

# ╔═╡ c9496d2a-931b-11eb-0228-5f08b4e1ff9f
md"""
##### Normalizing imputed values
Interestingly, when we smooth, we lose our "discrete" count values thus a negative binomial is no longer applicable. Let's try to fit Gamma distributions to our gene distributions and see how we do
"""

# ╔═╡ e75442f4-931b-11eb-0a5f-91e8c58ab47e
begin
	function clamp(value, lo, hi)
		if value < lo
			return lo
		elseif value > hi
			return hi
		else
			return value
		end
	end
	
	function gamma_loss(x⃗, z⃗, β̄, δβ¯²)
		function f(Θ)
			α, β, γ = Θ

			M = (exp(α+β*z) for z ∈ z⃗)
			Z = (loggamma(μ/γ)+(μ/γ)*log(γ) for μ ∈ M)  

			return -sum(-z + (μ/γ-1)*log(x)-x/γ for (z,μ,x) ∈ zip(Z,M,x⃗)) + 0.5*δβ¯²*(β-β̄)^2
		end
		
		return f
	end
	
	function fit_continuous1(x, z, β̄, δβ¯²)
		μ  = mean(x)
		Θ₀ = [
			log(μ),
			β̄,
			μ^2 / (var(x)-μ),
		]

		if Θ₀[end] < 0 || isinf(Θ₀[end])
			Θ₀[end] = 1
		end

		loss = 	TwiceDifferentiable(
			gamma_loss(x, z, β̄, δβ¯²),
			Θ₀;
			autodiff=:forward
		)

		constraint = TwiceDifferentiableConstraints(
			[-∞, -∞, 0],
			[+∞, +∞, +∞],
		)

		soln = optimize(loss, constraint, Θ₀, IPNewton())

		Θ̂  = Optim.minimizer(soln)
		Ê  = Optim.minimum(soln)
		δΘ̂ = diag(inv(hessian!(loss, Θ̂)))

		# pearson residuals
		α̂, β̂, γ̂ = Θ̂
		μ̂ = exp.(α̂ .+ β̂*z)
		σ̂ = .√(μ̂ .+ μ̂.^2 ./ γ̂)

		# cdf
		k   = μ̂ ./ γ̂
		cdf = GSL.sf_gamma_inc_P.(k, x./γ̂)

		# gaussian residuals
		ρ = erfinv.(clamp.(2 .*cdf .- 1,-1,1))
		ρ[isinf.(ρ)] = 10*sign.(ρ[isinf.(ρ)])

		return (
			parameters=Θ̂, 
			uncertainty=δΘ̂, 
			likelihood=Ê,
			trend=μ̂,
			cdf=cdf,
			residuals=ρ,
		)
	end
	
	function fit_continuous(data; β̄=1e0, δβ¯²=1e1)
		z = log.(vec(mean(data, dims=1)))
		χ = log.(vec(mean(data, dims=2)))

		fits = [begin 
					@show i 
					fit_continuous1(vec(data[i,:]), z, β̄, δβ¯²) 
				end for i ∈ 1:size(data,1)]

		return vcat((fit.residuals' for fit ∈ fits)...),
			(
				likelihood  = map((f)->f.likelihood,  fits),

				α  = map((f)->f.parameters[1],  fits),
				β  = map((f)->f.parameters[2],  fits),
				γ  = map((f)->f.parameters[3],  fits),

				δα = map((f)->f.uncertainty[1], fits),
				δβ = map((f)->f.uncertainty[2], fits),
				δγ = map((f)->f.uncertainty[3], fits),

				μ̂   = map((f)->f.trend,  fits),
				cdf = map((f)->f.cdf,  fits),
			
				χ = vec(mean(data, dims=2)),
				M = vec(maximum(data, dims=2))
			)
		end
end

# ╔═╡ 94fa4c96-931c-11eb-1a1e-556bb10223f5
Sᵧ, pᵧ = let
	S = seq.data ./ sum(seq.data,dims=1)
	Ñ = (S*Kₑ) .* sum(seq.data,dims=1)
	fit_continuous(Ñ)
end

# ╔═╡ 4ff1b8b4-9321-11eb-249b-1f35bc1facce
let
	p = scatter(pᵧ.χ, pᵧ.α, 
		alpha=0.1, 
		xscale=:log10, 
		marker_z=log10.(pᵧ.δα), 
		label=false
	)
	xaxis!("expression / cell")
	yaxis!("estimated α")
	p
end

# ╔═╡ 5a9982a6-9321-11eb-37a5-0be0a5b05d42
let
	p = scatter(pᵧ.χ, pᵧ.β, 
		alpha=0.1, 
		xscale=:log10, 
		marker_z=log10.(pᵧ.δβ), 
		label=false
	)
	xaxis!("expression / cell")
	yaxis!("estimated β")
	p
end

# ╔═╡ 7220bdb0-9321-11eb-2a89-79bcaa28726a
let
	p = scatter(pᵧ.χ, pᵧ.γ, 
		alpha=0.1, 
		xscale=:log10,
		yscale=:log10, 
		marker_z=log10.(pᵧ.δγ), 
		label=false
	)
	xaxis!("expression / cell")
	yaxis!("estimated γ")
	p
end

# ╔═╡ 87d39e64-9321-11eb-0170-834e21b45cc4
let
	cm = ColorSchemes.inferno
	χ  = log10.(pᵧ.χ)
	χ  = (χ .- minimum(χ)) ./ (maximum(χ) - minimum(χ))
	
	p = cdfplot(pᵧ.cdf[1], color=get(cm,χ[1]), alpha=0.01, label="")
	for i ∈ 2:5:length(pᵧ.cdf)
		cdfplot!(pᵧ.cdf[i], color=get(cm,χ[i]), alpha=0.01, label="")
	end
	
	plot!(0:1, 0:1, linestyle=:dashdot, color=:coral2, label="ideal", legend=:bottomright, linewidth=2)
	p
end

# ╔═╡ bb4aac90-9323-11eb-3562-afdd70610e24
let
	F = svd(Sᵧ)
	cdfplot(F.S, linewidth=2, xscale=:log10, label="empirical")
	vline!([sqrt(size(Sᵧ,1)) + sqrt(size(Sᵧ,2))], linestyle=:dashdot, linewidth=2, label="MP maximum")
	
	k = sum(F.S .> (sqrt(size(Sᵧ,1)) + sqrt(size(Sᵧ,2))))

	xaxis!("singular values")
	yaxis!("CDF")
	title!("rank ≈ $k")
end

# ╔═╡ 45ada9de-9325-11eb-1450-793727639203
S̃ᵧ = let d = 40
	F = svd(Sᵧ);
	F.U[:,1:d]*Diagonal(F.S[1:d])*F.Vt[1:d,:]
end;

# ╔═╡ 5c553b5e-9325-11eb-3f4b-e12c3c1c743b
Gᵧ = PointCloud.geodesics(S̃ᵧ, 12); size(Gᵧ)

# ╔═╡ 14462c08-933c-11eb-3369-3d081358bb35
import PyPlot

# ╔═╡ 17e1e152-933c-11eb-335b-6f2b072168ce
PyPlot.clf(); PyPlot.matshow(Gᵧ); PyPlot.gcf()

# ╔═╡ 58dd7382-9326-11eb-09d6-15251ddbb0bd
let
	ρ, Rs = PointCloud.scaling(Gᵧ, 1000)

	p = plot(Rs, ρ', alpha=0.03, color=:cyan3, label="")
	plot!(Rs, mean(ρ, dims=1)', linewidth=5, alpha=1, color=:cyan3, label="mean")

	xaxis!("radius", :log10)
	yaxis!("number of points", :log10)

	plot!(Rs, 4e-3*Rs.^3, linestyle=:dashdot, linewidth=2, alpha=1, color=:coral3, label="αR³", legend=:bottomright)
	plot!(Rs, 4e-2*Rs.^2, linestyle=:dashdot, linewidth=2, alpha=1, color=:coral3, label="αR²", legend=:bottomright)

	p
end

# ╔═╡ c31f47ee-9334-11eb-0fa1-1be19bf991ce
ν, ω = load("params.jld2", "ν", "ω")

# ╔═╡ d7e8e7a6-9326-11eb-0a3d-e918e72dac08
ψ, embryo, db, Φ = Inference.inversion(Sᵧ, seq.gene; ν=ν, ω=ω);

# ╔═╡ b2104258-9327-11eb-2151-69959f436b69
Ψᵣ = ψ(0.1)*size(Sᵧ,2);

# ╔═╡ 5b7847d6-9329-11eb-0269-dbbcf2d1e563
md"""
##### AP Position
"""

# ╔═╡ 18b52476-9326-11eb-26ac-8ff0d7afe21e
let
	ξ = PointCloud.mds(Gᵧ.^2, 3)
	AP = embryo[:,1]
	WGLMakie.scatter(Ψᵣ*ξ[:,1], Ψᵣ*ξ[:,2], Ψᵣ*ξ[:,3], color=AP, markersize=100)
end

# ╔═╡ 66700b9c-9329-11eb-1cde-87750e3652af
md"""
##### DV Position
"""

# ╔═╡ 1fcae9d8-9328-11eb-13fb-6935aaa65434
let
	ξ  = PointCloud.mds(Gᵧ.^2, 3)
	DV = embryo[:,2]#atan.(embryo[:,2],embryo[:,3])

	WGLMakie.scatter(Ψᵣ*ξ[:,1], Ψᵣ*ξ[:,2], Ψᵣ*ξ[:,3], color=DV, markersize=100)
end

# ╔═╡ f97f0ac6-9335-11eb-3051-336f0f48781f
ξ = PointCloud.mds(Gᵧ.^2, 3);

# ╔═╡ be47ae4c-9335-11eb-1deb-1732ce57ed4b
seq.gene[sortperm([cor(vec(Sᵧ[i,:]), ξ[:,1]) for i in 1:size(seq,1)])][1:50]

# ╔═╡ 0838eadc-9336-11eb-0079-23c7230a7bbc
seq.gene[sortperm([cor(vec(Sᵧ[i,:]), ξ[:,1]) for i in 1:size(seq,1)])][end-50:end]

# ╔═╡ 1a3fbf94-9336-11eb-2e48-35f09a2f14d5
seq.gene[sortperm([cor(vec(Sᵧ[i,:]), ξ[:,2]) for i in 1:size(seq,1)])][1:50]

# ╔═╡ 14aa52a6-9336-11eb-1a3d-a97d2d2dbe07
seq.gene[sortperm([cor(vec(Sᵧ[i,:]), ξ[:,2]) for i in 1:size(seq,1)])][end-50:end]

# ╔═╡ 55715968-9336-11eb-1845-a583df1166f6
seq.gene[sortperm([cor(vec(Sᵧ[i,:]), ξ[:,3]) for i in 1:size(seq,1)])][1:50]

# ╔═╡ 58255396-9336-11eb-23e5-67d8002b5595
seq.gene[sortperm([cor(vec(Sᵧ[i,:]), ξ[:,3]) for i in 1:size(seq,1)])][end-50:end]

# ╔═╡ a958c94a-932d-11eb-247c-1b44b35cf02f
GENE2 = findfirst(seq.gene .== "eve")

# ╔═╡ 497aee7c-932d-11eb-07fe-3b41d04c90f8
let
	WGLMakie.scatter(-embryo[:,1], embryo[:,2], embryo[:,3], color=Ψᵣ*Sᵧ[GENE2,:], markersize=2000)
end

# ╔═╡ 4fcb204a-933b-11eb-3797-571078415e34
let
	WGLMakie.scatter(-embryo[:,1], embryo[:,2], embryo[:,3], color=vec(Ψᵣ[:,1100]), markersize=2000)
end

# ╔═╡ d2339cb4-932a-11eb-293f-d35764e7b8f7
md"""
#### Conclusion
I think this works great. The pipeline in broad strokes is:
  1. Fit highly expressed genes to negative binomial.
  2. Use estimates to obtain prior for γ.
  3. Fit all genes to negative binomial with prior from before. 
  4. Use overdispersion factor γ in the variance rescaling equation to set all row and column variances to 1. 
  5. Find the gap in the spectrum, i.e. the rank of the original count matrix.
  6. Use the obtained low-dimensional manifold to estimate intercellular distances.
  7. Use distances in Gaussian kernel to lightly smooth the original raw data. 
  8. Fit smoothed data to Γ distribution.
  9. Transform into standard normal variables.
"""

# ╔═╡ 8e009626-9316-11eb-039e-fb1183e60421
md"""
#### Correlation
"""

# ╔═╡ 93f9188a-9316-11eb-1c35-2d93a1936277
Dₚ = let S = S̃ₐ
	D = zeros(size(S,2), size(S,2))
	for c₁ ∈ 1:size(S,2)
		for c₂ ∈ (c₁+1):size(S,2)
			D[c₁,c₂] = D[c₂, c₁] = 1 - cor(S[:,c₁], S[:,c₂])
		end
	end
	D
end;

# ╔═╡ 59d58e3c-9317-11eb-18c2-a54f3bfcb976
Gₚ = PointCloud.geodesics(S̃ₐ, 6; D=Dₚ); size(Gₚ)

# ╔═╡ f05f1616-9317-11eb-1a84-1b60e7dae579
let
	ρ, Rs = PointCloud.scaling(Gₚ, 1000)

	p = plot(Rs, ρ', alpha=0.03, color=:cyan3, label="")
	plot!(Rs, mean(ρ, dims=1)', linewidth=5, alpha=1, color=:cyan3, label="mean")

	xaxis!("radius", :log10)
	yaxis!("number of points", :log10)

	plot!(Rs, 1e4*Rs.^3, linestyle=:dashdot, linewidth=2, alpha=1, color=:coral3, label="αR³", legend=:bottomright)

	p
end

# ╔═╡ 5271d60c-931a-11eb-1bbb-c1bddc042136
Kₚ = let
	σ = .05*mean(Gₚ[:])
	K = exp.(-(Gₚ./σ).^2)
	u, v = sinkhorn(K; r=1, c=1)
	Diagonal(u) * K * Diagonal(v)
end;

# ╔═╡ 4fd9ff7c-9318-11eb-197b-9f916f33f983
let
	ξ = PointCloud.mds(Gₚ.^2, 3)
	WGLMakie.scatter(ξ[:,1], ξ[:,2], ξ[:,3], color=Kₚ*Sᵪ[GENE,:], markersize=10)
end

# ╔═╡ 83fe3a24-931a-11eb-0d09-e3eed2b09138
let
	Ñ = seq.data*Kₚ
	c = [ cor(vec(Ñ[i,:]), vec(seq[i,:])) for i in 1:size(Ñ,1) ]
	
	cdfplot(c, linewidth=2, label="")

	xaxis!("correlation before/after smoothing")
	yaxis!("CDF")
end

# ╔═╡ b77cf672-92f4-11eb-0301-8d060297d6f3
md"""
#### Wasserstein (prototype)
"""

# ╔═╡ f79db462-9283-11eb-1d3b-c19f6d9dee03
function cost_matrix(data)
	C = zeros(size(data,1), size(data,1))
	cos(x,y) = x⋅y / (norm(x)*norm(y))
	for i ∈ 1:size(data,1)
		@show i
		for j ∈ (i+1):size(data,1)
			C[i,j] = C[j,i] = 1 - cos(vec(data[i,:]),vec(data[j,:]))
		end
	end
	
	return C
end

# ╔═╡ 2d03d7b4-927f-11eb-0e2c-3b8ea5c52055
function pairwise_dists(data, cost; β=.1)
	D = zeros(size(data,2), size(data,2))
	
	# initialize cost matrix	
	H = exp.(β*cost .- 1)
	
	# initialize compute kernel
	kernel = (a, b) -> begin
		u, v = sinkhorn(H; r=a, c=b)
		p = Diagonal(u)*H*Diagonal(v)
		
		return sum(cost.*p) - sum(p.*log(p))/β
	end
	
	# compute
	Threads.@threads for i ∈ 1:size(data,2)
		@show i
		for j ∈ i:size(data,2)
			@show j
			D[i,j] = D[j,i] = kernel(data[:,i], data[:,j])
		end
	end
	
	D = D - .5*(diagonal(D) .+ diagonal(D)')
	
	return D
end

# ╔═╡ Cell order:
# ╟─2466c2a4-90c7-11eb-3f9c-5b87a7a35bb6
# ╟─fc2b03f0-924b-11eb-0f20-45edefca4b76
# ╟─969b3f50-90bb-11eb-2b67-c784d20c0eb2
# ╟─be981c3a-90bb-11eb-3216-6bed955446f5
# ╟─e84220a8-90bb-11eb-2fb8-adb6c87c2faa
# ╟─ce78d71a-917a-11eb-3cdd-b15aad75d147
# ╟─0bb97860-917f-11eb-3dd7-cfd0c7d890cd
# ╟─f972b674-9264-11eb-0a1c-0774ce2527e7
# ╟─9594926e-91d0-11eb-22de-bdfe3290b19b
# ╟─d7e28c3e-9246-11eb-0fd3-af6f94ea8562
# ╠═ca2dba38-9184-11eb-1793-f588473daad1
# ╟─4f65301e-9186-11eb-1faa-71977e8fb097
# ╟─bf8b0edc-9247-11eb-0ed5-7b1d16e00fc4
# ╟─f387130c-924f-11eb-2ada-794dfbf4d30a
# ╠═eed136bc-924d-11eb-3e3a-374d21772e4b
# ╠═7f38ceba-9253-11eb-000f-25045179e841
# ╟─b14e4f0e-9253-11eb-171a-053dcc942240
# ╠═cbcbb038-9255-11eb-0fc3-3ba4d95cee62
# ╟─b8bb97aa-9256-11eb-253f-a16885888c5f
# ╟─5b25177c-925d-11eb-1aec-f52c8c52ec93
# ╟─c65e8a86-9259-11eb-29bb-3bf5c089746f
# ╟─5ebfd944-9262-11eb-3740-37ba8930e1c6
# ╠═b563c86c-9264-11eb-01c2-bb42d74c3e69
# ╠═c423eca6-9264-11eb-375f-953e02fc7ec4
# ╟─fb056230-9265-11eb-1a98-33c234a0f959
# ╟─c8c4e6b4-9266-11eb-05e6-917b82a580ab
# ╟─622247ba-9268-11eb-3c0b-07f9cd3c6236
# ╟─d9880e94-92f6-11eb-3f1e-cf4462c3b89a
# ╟─17395b62-9272-11eb-0237-430f2e6499d6
# ╟─3886d34c-9279-11eb-31e6-0fd49a2694aa
# ╠═3d3a37d6-927d-11eb-16b3-e74d10851013
# ╠═8472a6a6-927d-11eb-0992-897698a13544
# ╠═a82fe0da-926a-11eb-063b-f70bd587a789
# ╟─a79aaeb0-926b-11eb-32ea-2bc1ead29909
# ╟─f53061cc-92f4-11eb-229b-b9b501c6cab8
# ╟─f5ef8128-90bb-11eb-1f4b-053ed41f5038
# ╠═f6f99ff4-92f5-11eb-2577-e51ab1cacfa6
# ╟─65224940-92f6-11eb-3045-4fc4b7b29a6c
# ╟─992dd34e-92f6-11eb-2642-bb9862148733
# ╟─a453ad84-92f6-11eb-0350-d1fbfbbbfda0
# ╟─f9d549ac-92f6-11eb-306a-77bf90c8eb33
# ╟─94b6d52a-92f8-11eb-226f-27e28f4a4d4b
# ╟─4f96560a-92fd-11eb-19f7-7b11f0ee46bf
# ╟─1f2bdf42-92ff-11eb-0e1f-e7c4e1e78058
# ╟─bc1990b6-92f7-11eb-0015-310aefc5041e
# ╟─9f56e6fc-9303-11eb-2bdd-5da078b4d9c3
# ╠═37afad28-9304-11eb-2f4d-8397fca7bb99
# ╠═d2e5b674-9305-11eb-1254-5b9fa030e8ee
# ╟─70fc4538-9306-11eb-1509-3dfa1034d8aa
# ╟─861ca2b6-9306-11eb-311c-69c6984faa26
# ╟─97ded136-9306-11eb-019b-636b739a61d6
# ╟─e05d2a98-9306-11eb-252e-5baf6fc59f8f
# ╟─bdffde54-9307-11eb-2ccb-ed48777f28f8
# ╟─91742bde-9309-11eb-2acc-836cc1ab1aee
# ╟─df6e73de-9309-11eb-3bd3-3f9f511744cf
# ╟─071f2c26-930c-11eb-2745-93cb2001e76b
# ╟─c8227e0e-9326-11eb-2724-cb588170c7c2
# ╟─8ffaa54e-930b-11eb-2f1f-9907008b76d2
# ╠═dd9986f4-930f-11eb-33b8-67dbc4d1d087
# ╠═3c979aa0-930c-11eb-3e6e-9bdf7f3029b5
# ╟─ed6333c6-930c-11eb-25c0-c551592746e0
# ╟─76b9c454-9316-11eb-1548-bdef70d532d6
# ╟─585bed74-9319-11eb-2854-9d74e1d592c2
# ╟─7cf6be2e-9315-11eb-1cb1-396f2131908b
# ╟─91147112-9315-11eb-0a71-8ddfdb2497a6
# ╟─f7d6acc2-9314-11eb-03f3-f9f221ded27c
# ╠═e96bc054-9310-11eb-25ca-576caeb336b8
# ╟─7a3b0202-9311-11eb-101e-99fbc40c6633
# ╟─c9496d2a-931b-11eb-0228-5f08b4e1ff9f
# ╟─e75442f4-931b-11eb-0a5f-91e8c58ab47e
# ╠═94fa4c96-931c-11eb-1a1e-556bb10223f5
# ╟─4ff1b8b4-9321-11eb-249b-1f35bc1facce
# ╟─5a9982a6-9321-11eb-37a5-0be0a5b05d42
# ╟─7220bdb0-9321-11eb-2a89-79bcaa28726a
# ╟─87d39e64-9321-11eb-0170-834e21b45cc4
# ╟─bb4aac90-9323-11eb-3562-afdd70610e24
# ╟─45ada9de-9325-11eb-1450-793727639203
# ╠═5c553b5e-9325-11eb-3f4b-e12c3c1c743b
# ╠═14462c08-933c-11eb-3369-3d081358bb35
# ╠═17e1e152-933c-11eb-335b-6f2b072168ce
# ╟─58dd7382-9326-11eb-09d6-15251ddbb0bd
# ╠═b992e41a-9334-11eb-1919-87967d572a21
# ╠═c31f47ee-9334-11eb-0fa1-1be19bf991ce
# ╠═d7e8e7a6-9326-11eb-0a3d-e918e72dac08
# ╠═b2104258-9327-11eb-2151-69959f436b69
# ╟─5b7847d6-9329-11eb-0269-dbbcf2d1e563
# ╠═18b52476-9326-11eb-26ac-8ff0d7afe21e
# ╟─66700b9c-9329-11eb-1cde-87750e3652af
# ╠═1fcae9d8-9328-11eb-13fb-6935aaa65434
# ╠═f97f0ac6-9335-11eb-3051-336f0f48781f
# ╟─be47ae4c-9335-11eb-1deb-1732ce57ed4b
# ╟─0838eadc-9336-11eb-0079-23c7230a7bbc
# ╟─1a3fbf94-9336-11eb-2e48-35f09a2f14d5
# ╟─14aa52a6-9336-11eb-1a3d-a97d2d2dbe07
# ╟─55715968-9336-11eb-1845-a583df1166f6
# ╟─58255396-9336-11eb-23e5-67d8002b5595
# ╠═a958c94a-932d-11eb-247c-1b44b35cf02f
# ╟─497aee7c-932d-11eb-07fe-3b41d04c90f8
# ╠═4fcb204a-933b-11eb-3797-571078415e34
# ╠═d2339cb4-932a-11eb-293f-d35764e7b8f7
# ╟─8e009626-9316-11eb-039e-fb1183e60421
# ╟─93f9188a-9316-11eb-1c35-2d93a1936277
# ╟─59d58e3c-9317-11eb-18c2-a54f3bfcb976
# ╟─f05f1616-9317-11eb-1a84-1b60e7dae579
# ╟─4fd9ff7c-9318-11eb-197b-9f916f33f983
# ╠═5271d60c-931a-11eb-1bbb-c1bddc042136
# ╟─83fe3a24-931a-11eb-0d09-e3eed2b09138
# ╟─b77cf672-92f4-11eb-0301-8d060297d6f3
# ╟─f79db462-9283-11eb-1d3b-c19f6d9dee03
# ╟─2d03d7b4-927f-11eb-0e2c-3b8ea5c52055
