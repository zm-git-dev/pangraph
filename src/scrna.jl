module scRNA

using GSL
using Statistics, StatsBase, Distributions
using SpecialFunctions, Interpolations

using NMF, Optim, NLSolversBase, ForwardDiff

import Base: 
    size, 
    IndexStyle, getindex, setindex!,
    +, -, *, /, %, ^,
    ∪

import LinearAlgebra:
    svd, diag, Diagonal, AbstractMatrix

include("io.jl")
using .DataIO: read_mtx, read_barcodes, read_features

# include("filter.jl")
# using .DataFilter

export barcodes, genes

# ------------------------------------------------------------------------
# globals

BarcodeFile = "barcodes.tsv"
FeatureFile = "features.tsv"
CountMatrix = "matrix.mtx"

const ∞ = Inf
const FitType = NamedTuple{
        (
            :parameters, 
            :uncertainty, 
            :likelihood, 
            :trend, 
            :cdf,
            :residuals
        ),
        Tuple{
            Array{Float64,1},
            Array{Float64,1},
            Float64,
            Array{Float64,1},
            Array{Float64,1},
            Array{Float64,1}
        }
}

# ------------------------------------------------------------------------
# utility functions

barcode(name) = occursin("/",name) ? join(split(name,"/")[2:end], "/") : name

function trendline(x, y, n)
    l,r = log(minimum(x)), log(maximum(x))
    bp  = range(l, r, length=n+1)
    x₀  = Array{eltype(x),1}(undef, n)
    μ   = Array{eltype(y),1}(undef, n)
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

function clamp!(x, lo, hi)
    x[x .< lo] .= lo
    x[x .> hi] .= hi
    x
end

betainc(a,b,x) = GSL.sf_beta_inc(a,b,x)
gammainc(a,x)  = GSL.sf_gamma_inc_P(a,x)

# ------------------------------------------------------------------------
# type w/ (de)serialization

struct Count{T <: Real} <: AbstractArray{T,2}
    data :: Array{T,2}
    gene :: Array{AbstractString,1}
    cell :: Array{AbstractString,1}

    Count(data::Array{T,2}, gene, cell) where T <: Real= begin
        if length(gene) != size(data,1) 
            error("number of genes $(length(gene)) not equal to number of rows $(size(data,1))")
        end

        if length(cell) != size(data,2) 
            error("number of cells $(length(cell)) not equal to number of columns $(size(data,2))")
        end

        new{T}(data,gene,cell)
    end
end

Count(data::Array{T,1}, gene::S, cell) where {T <: Real, S <: AbstractString} = Count(reshape(data,1,length(data)), [gene], cell)

# ------------------------------------------------------------------------
# simple operators

# data access
genes(seq::Count) = seq.gene
cells(seq::Count) = seq.cell

ngenes(seq::Count) = size(seq,1)
ncells(seq::Count) = size(seq,2)

# arithmetic
+(seq::Count, x::Union{<:Number,<:AbstractMatrix}) = Count(seq.data+x, seq.gene, seq.cell)
-(seq::Count, x::Union{<:Number,<:AbstractMatrix}) = Count(seq.data-x, seq.gene, seq.cell)

*(seq::Count, x::Union{<:Number,<:AbstractMatrix}) = Count(seq.data*x, seq.gene, seq.cell)
/(seq::Count, x::Number) = Count(seq.data/x, seq.gene, seq.cell)
%(seq::Count, x::Number) = Count(seq.data%x, seq.gene, seq.cell)
^(seq::Count, x::Number) = Count(seq.data^x, seq.gene, seq.cell)

# ------------------------------------------------------------------------
# matrix interface

size(seq::Count) = size(seq.data)

# -- indexing

IndexStyle(seq::Count) = IndexCartesian()

# integer indexes
getindex(seq::Count, I::Vararg{Int,2}) = getindex(seq.data, I...)
getindex(seq::Count, I::AbstractArray{<:Integer}, J) = Count(getindex(seq.data,I,J),seq.gene[I],seq.cell[J]) 
getindex(seq::Count, I, J::AbstractArray{<:Integer}) = Count(getindex(seq.data,I,J),seq.gene[I],seq.cell[J]) 
getindex(seq::Count, I::AbstractArray{<:Integer}, J::AbstractArray{<:Integer}) = Count(getindex(seq.data,I,J),seq.gene[I],seq.cell[J]) 

setindex!(seq::Count, v, I::Vararg{Int,2}) = setindex!(seq.data, v, I...)

# string indexes
function getindex(seq::Count, I::Vararg{String,2})
    gene, batch = I
    row = findfirst(seq.gene .== gene)
    col = inbatch(seq, batch)
    return getindex(seq, row, col)
end

ArrayOrOne{T} = Union{<:T, AbstractArray{<:T,1}}

findrows(seq,G) = G isa AbstractArray ? [findfirst(seq.gene .== g) for g in G] : findfirst(seq.gene .== G)
findcols(seq,B) = B isa AbstractArray ? reduce((x,y) -> x .| y, inbatch(seq, b) for b ∈ B) : inbatch(seq, B)

function getindex(seq::Count, genes::ArrayOrOne{AbstractString}, batches::ArrayOrOne{AbstractString})
    row = findrows(seq,genes)
    col = findcols(seq,batches)

    return getindex(seq, row, col)
end

function getindex(seq::Count, genes::ArrayOrOne{<:AbstractString}, batches::Function)
    row = findrows(seq,genes)
    return getindex(seq, row, batches)
end

function getindex(seq::Count, genes::Function, batches::ArrayOrOne{<:AbstractString})
    col = findcols(seq,batches)
    return getindex(seq, genes, col)
end

# mixed indexes

function getindex(seq::Count, row::ArrayOrOne{Integer}, batches::ArrayOrOne{AbstractString})
    col = findcols(seq,batches)
    return getindex(seq, row, col)
end

function getindex(seq::Count, genes::ArrayOrOne{AbstractString}, col::ArrayOrOne{Integer})
    row = findrows(seq,genes)
    return getindex(seq, row, col)
end

# -- matrix operations

svd(seq::Count) = svd(seq.data)

# -- batch subselection

function batches(seq::Count) 
    all = map(cells(seq)) do id
        s = split(id, "/")
        return length(s) > 1 ? s[1] : nothing
    end
    all = filter(!isnothing, all)

    unique!(all)
    return all
end

inbatch(seq::Count, batch::AbstractString) = startswith.(cells(seq), batch*"/")

# -- gene subselection

locus(seq::Count, genes::AbstractString...)    = filter((i)->!isnothing(i), [findfirst(seq.gene .== g) for g in genes])
searchloci(seq::Count, prefix::AbstractString) = occursin.(prefix, seq.gene)

# ------------------------------------------------------------------------
# creation operators

function load(dir::AbstractString; batch=missing)
    !isdir(dir) && error("directory '$dir' not found")

    files = readdir(dir)

    BarcodeFile ∉ files && error("'$BarcodeFile' not found in directory '$dir'")
    FeatureFile ∉ files && error("'$FeatureFile' not found in directory '$dir'")
    CountMatrix ∉ files && error("'$CountMatrix' not found in directory '$dir'")

    data     = open(read_mtx,      "$dir/$CountMatrix")
    barcodes = open(read_barcodes, "$dir/$BarcodeFile")
    features = open(read_features, "$dir/$FeatureFile")

    length(barcodes) ≠ size(data,2) && error("number of barcodes $(length(barcodes)) ≠ number of columns $(size(counts,2)). check data")
    length(features) ≠ size(data,1) && error("number of features $(length(features)) ≠ number of rows $(size(counts,1)). check data")

    # TODO: accept sparse inputs?
    prepend = ismissing(batch) ? (bc) -> bc : (bc) -> batch * "/" * bc
    return Count(Matrix(data),features,map(prepend,barcodes))
end

matchperm(a, b) = findfirst.(isequal.(a), (b,))

function ∪(seq₁::Count{T}, seq₂::Count{S}) where {T <: Real, S <: Real}
    T₀ = T ≠ S ? promote_rule(T,S) : T
    matches = matchperm(genes(seq₂), genes(seq₁)) # match 2 -> 1

    newgenes = [gene for (gene,match) ∈ zip(genes(seq₂),matches) if isnothing(match) ]
    features = [genes(seq₁) ; newgenes]
    barcodes = [cells(seq₁) ; cells(seq₂)]

    n = 1
    δ = ngenes(seq₁)
    for (i,m) ∈ enumerate(matches)
        if isnothing(m)
            matches[i] = n + δ
            n += 1
        end
    end

    data = [hcat(seq₁.data, zeros(T₀,ngenes(seq₁),ncells(seq₂)));
            zeros(T₀,length(newgenes),ncells(seq₁)+ncells(seq₂))]

    δ = ncells(seq₁)
    for i ∈ 1:ncells(seq₂)
        data[matches,i+δ] = seq₂[:,i]
    end

    return Count(data,features,barcodes)
end

# NOTE: different merging strategy
#       only common genes (intersection) are kept
function ∩(seq₁::Count{T}, seq₂::Count{S}) where {T <: Real, S <: Real}
    error("need to implement")
end

# ------------------------------------------------------------------------
# filtering

function filtergene(f, seq::Count)
    ι = [f(seq[i,:], seq.gene[i]) for i in 1:ngenes(seq)]
    return seq[ι,:]
end

function filtercell(f, seq::Count)
    ι = [f(seq[:,i], seq.cell[i]) for i in 1:ncells(seq)]
    return seq[:,ι]
end

# ------------------------------------------------------------------------
# normalization

function negbinom1_loss(x⃗, ḡ, β̄::Float64, δβ¯²::Float64)
	function f(Θ)
		α, β, γ = Θ
		
		Mu = (exp(+α + β*g) for g ∈ ḡ)
        G₁ = (loggamma(x+γ*μ) - loggamma(x+1) - loggamma(γ*μ) for (x,μ) ∈ zip(x⃗,Mu))
        G₂ = (x*log(γ) - (x+γ*μ)*log(1+γ) for (x,μ) ∈ zip(x⃗,Mu) )

        return -mean(g₁+g₂ for (g₁,g₂) ∈ zip(G₁,G₂)) + 0.5*δβ¯²*(β-β̄)^2
	end
	
	return f
end

function negbinom2_loss(x⃗, ḡ, β̄::Float64, δβ¯²::Float64)
	function f(Θ)
		α, β, γ = Θ
		
		G₁ = (loggamma(x+γ) - loggamma(x+1) - loggamma(γ) for x ∈ x⃗)
		G₂ = (x*(α + β*g)-(x+γ)*log(exp(α+β*g) + γ) for (x,g) ∈ zip(x⃗,ḡ))

        return -mean(g₁+g₂+γ*log(γ) for (g₁,g₂) ∈ zip(G₁,G₂)) + 0.5*δβ¯²*(β-β̄)^2
	end
	
	function ∂f!(∂Θ, Θ)
		α, β, γ = Θ
		
		Mu 	 = (exp(+α + β*g) for g ∈ ḡ)
		Mu¯¹ = (exp(-α - β*g) for g ∈ ḡ)

		G₁ = (x-((γ+x)/(1+γ*μ¯¹)) for (x,μ¯¹) ∈ zip(x⃗,Mu¯¹))
		G₂ = (digamma(x+γ)-digamma(γ)+log(γ)+1-log(γ+μ)-(x+γ)/(γ+μ) for (x,μ) ∈ zip(x⃗,Mu))

		∂Θ[1] = -mean(G₁)
        ∂Θ[2] = -mean(g₁*g for (g₁,g) ∈ zip(G₁,ḡ)) + δβ¯²*(β-β̄)
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
		∂²Θ[2,2] = mean(g₁*g^2 for (g₁,g) ∈ zip(G₁,ḡ)) + δβ¯²
		
		# γ row/column
		∂²Θ[3,3] = -mean(G₄)
		∂²Θ[3,1] = ∂²Θ[1,3] = -mean(G₃)
		∂²Θ[3,2] = ∂²Θ[2,3] = -mean(g₃*g for (g₃, g) ∈ zip(G₃,ḡ))
	end
	
	return f, ∂f!, ∂²f!
end

function gamma_loss(x⃗, ḡ, β̄::Float64, δβ¯²::Float64)
	function f(Θ)
		α, β, γ = Θ
		
        M = (exp(α+β*g) for g ∈ ḡ)
        Z = (loggamma(μ/γ)+(μ/γ)*log(γ) for μ ∈ M)  

        return -mean(-z + (μ/γ-1)*log(x)-x/γ for (z,μ,x) ∈ zip(Z,M,x⃗)) + 0.5*δβ¯²*(β-β̄)^2
	end

    # TODO: write out gradients by hand?
    return f
end


# Fits cell-specific parameters / gene
#
# input parameters are:
# x    => vector of genes to fit
# ḡ    => average gene expression for given cell
# β̄    => mean of prior on β
# δβ¯² => mean of prior of β
function fit_continuous(x, ḡ, β̄, δβ¯²)::FitType
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
        gamma_loss(x, ḡ, β̄, δβ¯²),
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
	μ̂ = exp.(α̂ .+ β̂*ḡ)
	σ̂ = .√(μ̂ .+ μ̂.^2 ./ γ̂)

    # cdf
    k   = μ̂ ./ γ̂
    cdf = gammainc.(k, x./γ̂)

    # gaussian residuals
    ρ = erfinv.(2 .*cdf .- 1)
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

function fit_discrete(x, ḡ, β̄, δβ¯²)::FitType
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
        negbinom2_loss(x, ḡ, β̄, δβ¯²)...,
		Θ₀
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
	μ̂ = exp.(α̂ .+ β̂*ḡ)
	σ̂ = .√(μ̂ .+ μ̂.^2 ./ γ̂)
    ρ = (x .- μ̂) ./ σ̂

    # cdf
    p = μ̂./(μ̂.+γ̂)
    p[p .< 0] .= 0
    p[p .> 1] .= 1
    cdf = 1 .- betainc.(x.+1.0, γ̂, p)

	return (
		parameters=Θ̂, 
		uncertainty=δΘ̂, 
		likelihood=Ê,
		trend=μ̂,
        cdf=cdf,
		residuals=ρ,
	)
end


# input parameters are:
# β̄    => mean of prior on β₁ (cell-specific count)
# δβ¯² => uncertainty of prior on β₁ (cell-specific count)
function normalize(seq::Count; algo=:continuous, opt=:multmse, β̄=1.0, δβ¯²=1e-2, k=500)
    ḡ   = log.(vec(mean(seq, dims=1)))
    est = if algo == :continuous
              d = float.(vec(sum(seq.data,dims=1)))
              if opt == :simple # just fit the fractional expression to a Gamma distribution
                  float.(seq.data) * Diagonal(1 ./ d)
              else
                  # model = NMF(k, init="nndsvda", verbose=2)
                  # W = model.fit_transform(float.(seq.data) * Diagonal(1 ./ d))
                  # H = model.components_
                  # (W*H)*Diagonal(d)
                  r = nnmf(
                            float.(seq.data) * Diagonal(1 ./ d),
                            k, 
                            init=:nndsvdar,
                            alg=opt, 
                            tol=1e-4,
                            maxiter=200,
                            verbose=true
                  )
                  (r.W*r.H)*Diagonal(d)
              end
          elseif algo == :discrete
              seq
          else
              error("expected algo to be either 'discrete' or 'continuous', received '$algo'")
          end

    fits = Array{FitType,1}(undef, ngenes(seq))
    if algo == :continuous
        Threads.@threads for i ∈ 1:ngenes(seq)
            fits[i] = fit_continuous(vec(est[i,:]), ḡ, β̄, δβ¯²)
        end
    else # already checked that algo is one of two options
        Threads.@threads for i ∈ 1:ngenes(seq)
            fits[i] = fit_discrete(vec(est[i,:]), ḡ, β̄, δβ¯²)
        end
    end

    return Count(vcat((fit.residuals' for fit ∈ fits)...), seq.gene, seq.cell),
        (
            likelihood  = map((f)->f.likelihood,  fits),

            α  = map((f)->f.parameters[1],  fits),
            β  = map((f)->f.parameters[2],  fits),
            γ  = map((f)->f.parameters[3],  fits),

            δα = map((f)->f.uncertainty[1], fits),
            δβ = map((f)->f.uncertainty[2], fits),
            δγ = map((f)->f.uncertainty[3], fits),

            raw = [vec(seq.data[i,:]) for i in 1:size(seq.data,1)],
            cdf = map((f)->f.cdf, fits),

            μ̂ = map((f)->f.trend,  fits),
            χ = vec(mean(est, dims=2)),
            M = vec(maximum(est, dims=2))
        )
end

function bisect(seq::Count)
    population(cell)  = reduce(vcat, [vec([g for _ in 1:n]) for (g,n) in enumerate(cell)])
    put!(cell, index) = for i ∈ index cell[i] += 1 end

    subcell₁ = zeros(eltype(seq.data), size(seq.data))
    subcell₂ = zeros(eltype(seq.data), size(seq.data))

    for c in 1:size(seq,2)
        N = sum(seq.data[:,c])
        put!(view(subcell₁,:,c), sample(population(seq.data[:,c]), N÷2; replace=false))
        subcell₂[:,c] = seq.data[:,c] - subcell₁[:,c]
    end

    count(data) = Count(data, seq.gene, seq.cell)
    return count(subcell₁), count(subcell₂)
end

function filterbarcodes(seq::Count)
    barcodes = open("$ROOT/barcodes") do io
        Set([line for line ∈ eachline(io)])
    end

    seq = filtercell(seq) do _, name
        barcode(name) ∈ barcodes
    end

    return seq
end

# ------------------------------------------------------------------------
# synthetic data generation

# TODO: gamma distribution?
#       correlated genes?
function generate(ngene, ncell; ρ=(α=Gamma(0.25,2), β=Normal(1,.01), γ=Gamma(3,3)))
    seq = zeros(Int, ngene, ncell)
    cdf = zeros(Float64, ngene, ncell)

    z = log.(rand(Gamma(5,1), ncell))

    α = log.(rand(ρ.α, ngene))
    β = rand(ρ.β, ngene)
    γ = rand(ρ.γ, ngene)

    for g ∈ 1:ngene
        μ⃗ = exp.(α[g] .+ β[g].*z)
        for (c, μ) ∈ enumerate(μ⃗)
            λ = rand(Gamma(γ[g], μ/γ[g]),1)[1]

            seq[g,c] = rand(Poisson(λ),1)[1]
        end
        p⃗ = μ⃗./(μ⃗.+γ[g])
        p⃗[p⃗ .< 0] .= 0
        p⃗[p⃗ .> 1] .= 1
        cdf[g,:] = 1 .- betainc.(vec(seq[g,:]).+1.0, γ[g], p⃗)

        #=
        for (c,p) ∈ enumerate(p⃗) 
            pmf(x) = exp.(loggamma.(x .+ γ[g]) .- loggamma.(x .+ 1) .- loggamma(γ[g]) + x.*log.(p) .+ γ[g].*log.(1 .- p))
            F(x) = sum(pmf(y) for y ∈ 0:x)
            cdf[g,c] = F(seq[g,c])
        end
        =#
    end

    # filter out any rows that don't express in at least one cell
    ι = vec(sum(seq,dims=2)) .> 0

    seq = seq[ι, :]
    cdf = cdf[ι, :]

    α = α[ι]
    β = β[ι]
    γ = γ[ι]

    return (
        data = Count(seq, [string(g) for g in 1:size(seq,1)], [string(c) for c in 1:size(seq,2)]),
        cdf  = cdf,
        α    = α,
        β    = β,
        γ    = γ,
        ḡ    = exp.(z),
   )
end

# ------------------------------------------------------------------------
# point of entry for testing

const ROOT = "/home/nolln/root/data/seqspace/raw"

process(seq) = begin
    markers  = (
        yolk = locus(seq, "sisA", "CG8129", "Corp", "CG8195", "CNT1", "ZnT77C"),
        pole = locus(seq, "pgc"),
        dvir = searchloci(seq, "Dvir_")
    )

    # remove pole and yolk cells
    state = "before"
    @show state, size(seq)
    seq = filtercell(seq) do cell, _
        (sum(cell[markers.yolk]) < 10 
      && sum(cell[markers.pole]) < 3 
      && sum(cell[markers.dvir]) < .1*sum(cell))
    end

    #=
    
    barcodes = open("$ROOT/barcodes") do io
        Set([line for line ∈ eachline(io)])
    end

    seq = filtercell(seq) do _, name
        barcode(name) ∈ barcodes
    end
    =#

    # keep only cells that average .1 expression across genes
    seq = filtercell(seq) do cell, _
        sum(cell) >= 1e-1*ngenes(seq)  
    end

    state = "after"
    @show state, size(seq)

    return seq
end

function test()
    # seq = reduce(∪, process(load("$ROOT/rep$r"; batch="rep$r")) for r ∈ [1,2,3,4,5,6,7])
    # seq = reduce(∪, process(load("$ROOT/rep$r")) for r ∈ [1,2,3,4,5,6,7])
    seq = reduce(∪, process(load("$ROOT/rep$r")) for r ∈ [3,4,5,6,7])
    seq = filtergene(seq) do gene, _
        sum(gene) >= 1e-2*length(gene) && maximum(gene) > 1
    end

    # return normalize(seq; β̄=1.0, δβ¯²=1e0, k=ncells(seq)÷2)
    
    # NOTE: This is kept for posterity! This is the normalization of DVEX paper
    #       They divide out by the column sum and scale by maximum cell depth
    #       Add a pseudo count and take log transform
    #       I recover the .85 correlation there! Solely a function of the pseudocount...
    
    seq = Count(log10.(maximum(sum(seq,dims=1)) .* seq ./ sum(seq, dims=1) .+ 1), seq.gene, seq.cell)
    return seq, nothing
end

end
