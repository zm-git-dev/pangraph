module DataFilter

using Markdown
using LinearAlgebra, Statistics, StatsBase

export MLE

# TODO: move somewhere else
md"""
## Data filtering
We need to scale the data so so that highly expressed genes and lowly expressed genes are equally weighted - i.e. focus on the variation between them not overall scale.
This is clear after looking at the principal components: dominated by inter-gene variation.

Define ``x_{in}`` to be the expression count for gene i and cell n

###### Idea: 
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
Hafemeister, C. & Satija, R. (2019) claims this overfits and thus put a heurestic prior on the parameters.
"""

# ------------------------------------------------------------------------
# globals

const ∞ = Inf
function clamp!(x,lo,hi)
    x[x < lo] = lo
    x[x > hi] = hi
    x
end

# ------------------------------------------------------------------------
# maximum likelihood estimation

module MLE
    using GSL, Optim, NLSolversBase
    using SpecialFunctions: erfinv

    incbeta(a,b,x) = GSL.sf_beta_inc(a,b,x)
    incgamma(a,x)  = GSL.sf_gamma_inc_P(a,x)

    # ---------------------------------------------------------------------
    # generalized linear models
    # Γ denotes parameters for prior distributions
    
    # the common formulation i.e. NB2
    function negative_binomial(x, z, Γ)
        (;β̄, δβ¯²,Γᵧ) = Γ

        if isnothing(Γᵧ)
            function f(Θ)
                α, β, γ = Θ
                L  = (loggamma(x+γ) 
                    - loggamma(x+1) 
                    - loggamma(γ) 
                    + x*(α+β*z)
                    + γ*log(γ)
                    - (x+γ)*log(exp(α+β*z)+γ) for (x,z) ∈ zip(x,z))

                return -sum(L) + 0.5*δβ¯²*(β-β̄)^2
            end
        else
            (;σ,ν,μ) = Γᵧ
            function f(Θ)
                α, β, γ = Θ
                L  = (loggamma(x+γ) 
                    - loggamma(x+1) 
                    - loggamma(γ) 
                    + x*(α+β*z)
                    + γ*log(γ)
                    - (x+γ)*log(exp(α+β*z)+γ) for (x,z) ∈ zip(x,z))

                return -sum(L) + 0.5*δβ¯²*(β-β̄)^2 + (abs(log(γ)-μ)/σ)^ν
            end
        end

        # TODO: 1st and 2nd derivatives

        μ  = mean(x)
        Θ₀ = [
            log(μ),
            β̄,
            var(x)/μ - 1,
        ]

        if Θ₀[end] < 0 || isinf(Θ₀[end])
            Θ₀[end] = 1
        end

        return (
            Θ₀         = Θ₀,
            loss       = TwiceDifferentiable(f, Θ₀; autodiff=:forward),
            constraint = TwiceDifferentiableConstraints([-∞,-∞,0],[+∞,+∞,+∞])
            residual   = (Θ) -> let
                α, β, γ = Θ
                μ = @. exp(α + β*z)
                σ̂ = @. √(μ̂ + μ̂^2/γ̂)
                return @. (x - μ) / σ
            end
            cumulative = (Θ) -> let
                α, β, γ = Θ
                μ = @. exp(α + β*z)
                p = @. μ / (μ + γ)
                return @. 1 - incbeta(x+1, γ, p)
            end
        )
    end

    function gamma(x, z, Γ) 
        (;β̄, δβ¯²) = Γ

        function f(Θ)
            α, β, γ = Θ

            μ = (exp(α+β*z) for z ∈ z)
            L = (-loggamma(μ/γ)
               -(μ/γ)*log(γ)
               +(μ/γ-1)*log(x)
               -x/γ for (μ,x) ∈ zip(μ,x))

            return -sum(L) + 0.5*δβ¯²*(β-β̄)^2
        end

        μ  = mean(x)
		Θ₀ = [
			log(μ),
			β̄,
			μ^2 / (var(x)-μ),
		]

        return (
            Θ₀         = Θ₀,
            loss       = TwiceDifferentiable(f, Θ₀; autodiff=:forward),
            constraint = TwiceDifferentiableConstraints([-∞,-∞,0],[+∞,+∞,+∞]),
            residual   = (Θ) -> let
                α, β, γ = Θ
                μ = @. exp(α + β*z)
                k = @. μ / γ
                Φ = @. incgamma(k, x/γ)
                ρ = @. erfinv(clamp!(2*cdf-1,-1,1))
                ρ[isinf.(ρ)] = 10*sign.(ρ[isinf.(ρ)])
                return ρ
            end
            cumulative = (Θ) -> let
                α, β, γ = Θ
                μ = @. exp(α + β*z)
                k = @. μ̂ / γ̂
                return @. incgamma(k, x/γ)
            end
        )
    end

    # ---------------------------------------------------------------------
    # simple distributions (no latent variables)
    
    # univariate gamma
    function gamma(x)
        f = (Θ) -> let
            k, θ = Θ
            return -sum((k-1)*log.(x) .- (x./θ) .- loggamma(k) .- k*log(θ))
        end

        μ  = mean(x)
        σ² = var(x)
        Θ₀ = [
            μ^2/σ²,
            σ²/μ,
        ]
	
        return (
            Θ₀         = Θ₀,
            loss       = TwiceDifferentiable(f, Θ₀; autodiff=:forward),
            constraint = TwiceDifferentiableConstraints([0,0],[+∞,+∞])
        )
    end

    # univariate lognormal
    function lognormal(x)
        f = (Θ) -> let
            μ, σ = Θ
            return sum(log(σ) .+ (log.(x) .- μ).^2 ./ (2*σ^2))
        end
        
        μ  = mean(log.(x))
        σ  = std(log.(x))
        Θ₀ = [
            μ,
            σ
        ]
        
        return (
            Θ₀         = Θ₀,
            loss       = TwiceDifferentiable(f, Θ₀; autodiff=:forward)
            constraint = TwiceDifferentiableConstraints([0,0],[+∞,+∞])
        )
    end

    # univariate generalized normal (variant 1 wikipedia)
    function gennormal(x)
        f = (Θ) -> let
            μ, σ, β = Θ
            return sum(loggamma(1/β) + log(σ) .+ (abs.(x - μ)./ σ).^β .- log(β))
        end
	
        μ  = mean(data)
        σ  = std(data)
        Θ₀ = [
            μ,
            σ,
            2
        ]
        
        return (
            Θ₀         = Θ₀,
            loss       = TwiceDifferentiable(f, Θ₀; autodiff=:forward),
            constraint = TwiceDifferentiableConstraints([0,0,0],[+∞,+∞,+∞]),
       )
    end

    # ---------------------------------------------------------------------
    # drivers

    # log link function is assumed
    function fitglm(model::Symbol, data; Γ=(β̄=1,δβ¯²=10), run=(x)->true)
        Σ = vec(mean(data, dims=1))

        foundmodel = try
            getfield(MLE, model)
        catch
            error("model '$model' not implemented")
        end
        modelfor(x) = foundmodel(x, log.(Σ), Γ)

        function fit(row, i)
            @show i
            model    = modelfor(row)
            estimate = optimize(model.loss, model.constraint, model.Θ₀, IPNewton())

            Θ  = Optim.minimizer(soln)
            E  = Optim.minimum(soln)
            δΘ = diag(inv(hessian!(model.loss, Θ)))
            return (
                likelihood  = E
                parameters  = Θ,
                uncertainty = δΘ,
                cumulative  = model.cumulative(Θ),
                residual    = model.residual(Θ),
            )
        end

        fits = [fit(row,i) for (i,row) ∈ enumerate(eachrow(data)) if run(row)]

        return vcat((fit.residuals' for fit ∈ fits)...),
            (
                likelihood  = map((f)->f.likelihood,  fits),

                # XXX: assumes particular form for parameters
                α  = map((f)->f.parameters[1],  fits),
                β  = map((f)->f.parameters[2],  fits),
                γ  = map((f)->f.parameters[3],  fits),

                δα = map((f)->f.uncertainty[1], fits),
                δβ = map((f)->f.uncertainty[2], fits),
                δγ = map((f)->f.uncertainty[3], fits),
            )

    end

    function fit(model, data)
        estimate = optimize(model.loss, model.constraint, model.Θ₀, IPNewton())
        return Optim.minimizer(estimate)
    end

end

end
