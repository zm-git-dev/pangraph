module DataFilter

using Markdown
using LinearAlgebra
using Statistics, StatsBase
using Optim, NLSolversBase, SpecialFunctions

export fit_glm, genes

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
"""

# ------------------------------------------------------------------------
# helper functions

# negative binomial(2)
# link function: μ = exp.(α .+ β.*ḡ)

function make_loss(x⃗, ḡ)
	function f(Θ)
		α, β, γ = Θ
		
		G₁ = (loggamma(x + γ) - loggamma.(x+1) .- loggamma(γ) for x ∈ x⃗)
		G₂ = (x*(α+β*g)-(x+γ)*log(exp(α + β*g) .+ γ) for (g,x) ∈ zip(ḡ, x⃗))

		return -mean(g₁+g₂+γ*log(γ) for (g₁,g₂) ∈ zip(G₁,G₂))
	end
	
	function ∂f!(∂Θ, Θ)
		α, β, γ = Θ
		
		Mu 	 = (exp(+α + β*g) for g ∈ ḡ)
		Mu¯¹ = (exp(-α - β*g) for g ∈ ḡ)

		G₁ = (x-((γ+x)/(1+γ*μ¯¹)) for (x,μ¯¹) ∈ zip(x⃗,Mu¯¹))
		G₂ = (digamma(x+γ)-digamma(γ)+log(γ)+1-log(γ+μ)-(x+γ)/(γ+μ) for (x,μ) ∈ zip(x⃗,Mu))

		∂Θ[1] = -mean(G₁)
		∂Θ[2] = -mean(g₁*g for (g₁,g) ∈ zip(G₁,ḡ))
		∂Θ[3] = -mean(G₂)
	end
	
	function ∂²f!(∂²Θ, Θ)
		α, β, γ = Θ
		
		Mu 	 = (exp(+α + β*g) for g ∈ ḡ)
		Mu¯¹ = (exp(-α - β*g) for g ∈ ḡ)

		G₁ = (γ*μ¯¹*(γ+x)/(1+γ*μ¯¹).^2 for (x,μ¯¹) ∈ zip(x⃗,Mu¯¹))
		G₂ = (μ+γ for μ ∈ Mu)
		G₃ = (μ*((γ+x)/g₂^2 - 1/g₂ ) for (x,μ,g₂) ∈ zip(x⃗,Mu,G₂))
		G₄ = (trigamma(x+γ)-trigamma(γ)+1/γ-2/g₂+(γ+x)/g₂^2 for (x,g₂) ∈ zip(x⃗,G₂))

		# α,β submatrix
		∂²Θ[1,1] = mean(G₁)
		∂²Θ[1,2] = ∂²Θ[2,1] = mean(g₁*g for (g₁,g) ∈ zip(G₁,ḡ))
		∂²Θ[2,2] = mean(g₁*g^2 for (g₁,g) ∈ zip(G₁,ḡ))
		
		# γ row/column
		∂²Θ[3,3] = -mean(G₄)
		∂²Θ[3,1] = ∂²Θ[1,3] = -mean(G₃)
		∂²Θ[3,2] = ∂²Θ[2,3] = -mean(g₃*g for (g₃, g) ∈ zip(G₃,ḡ))
	end
	
	return f, ∂f!, ∂²f!
end

# ------------------------------------------------------------------------
# main exports

function normalize(x, ḡ; check_grad=false)
	μ = mean(x)
	
	Θ₀ = [
		log(μ),
		0,
		μ^2 / (var(x)-μ),
	]
	
	if Θ₀[3] < 0 || isinf(Θ₀[3])
		Θ₀[3] = 1
	end
	
	loss = check_grad ? TwiceDifferentiable(
		make_loss(x, ḡ)[1],
		Θ₀;
		autodiff=:forward
	) : TwiceDifferentiable(
		make_loss(x, ḡ)...,
		Θ₀
	)
	
	if check_grad
		f, ∂f!, ∂²f! = make_loss(x, ḡ)
		∂, ∂² = zeros(3), zeros(3,3)

		∂f!(∂,Θ₀)
		∂²f!(∂²,Θ₀)

		@show gradient!(loss, Θ₀)
		@show ∂

		@show hessian!(loss, Θ₀)
		@show ∂²
	end

	constraint = TwiceDifferentiableConstraints(
		[-Inf, -Inf, 0],
		[+Inf, +Inf, +Inf],
	)

	soln = optimize(loss, constraint, Θ₀, IPNewton())
	# TODO: check if was successful
	
	Θ̂  = Optim.minimizer(soln)
	Ê  = Optim.minimum(soln)
	δΘ̂ = diag(inv(hessian!(loss, Θ̂)))
	
	# pearson residuals
	α̂, β̂, γ̂ = Θ̂
	μ̂ = exp.(α̂ .+ β̂*ḡ)
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

function fit_glm(data)
    T = NamedTuple{
        (
            :parameters, 
            :uncertainty, 
            :likelihood, 
            :trend, 
            :residuals
        ),
        Tuple{
            Array{Float64,1},
            Array{Float64,1},
            Float64,
            Array{Float64,1},
            Array{Float64,1}
        }
    }

    ḡ   = vec(mean(data, dims=1))
    fit = Array{T,1}(undef, size(data,1))

    Threads.@threads for i = 1:size(data,1)
        @show i
        fit[i] = normalize(data[i,:], ḡ)
    end

    return fit
end

function genes(data; min=1e-3)
	μ = vec(mean(data,dims=2))
    return Matrix(data[μ .> min,:])
end

end
