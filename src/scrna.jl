module scRNA

using Statistics, SpecialFunctions
using Optim, NLSolversBase

import Base: 
    size, 
    IndexStyle, getindex, setindex!,
    +, -, *, /, %, ^,
    ∪

import LinearAlgebra:
    svd, diag

include("io.jl")
using .DataIO: read_mtx, read_barcodes, read_features

include("filter.jl")
using .DataFilter

export barcodes, genes

# ------------------------------------------------------------------------
# globals

BarcodeFile = "barcodes.tsv"
FeatureFile = "features.tsv"
CountMatrix = "matrix.mtx"

const ∞ = Inf

# ------------------------------------------------------------------------
# utility functions

# ------------------------------------------------------------------------
# type w/ (de)serialization

struct Count{T <: Real} <: AbstractArray{T,2}
    data    :: Array{T,2}
    gene    :: Array{String,1}
    barcode :: Array{String,1}
end

Count(data::Array{T,2}, gene, barcode) where T <: Real = Count{T}(data,gene,barcode)

# ------------------------------------------------------------------------
# simple operators

# data access
genes(seq::Count) = seq.gene
cells(seq::Count) = seq.barcode

ngenes(seq::Count) = size(seq,1)
ncells(seq::Count) = size(seq,2)

# arithmetic
+(seq::Count, x::Number) = Count(seq.data+x, seq.gene, seq.barcode)
-(seq::Count, x::Number) = Count(seq.data-x, seq.gene, seq.barcode)

*(seq::Count, x::Number) = Count(seq.data*x, seq.gene, seq.barcode)
/(seq::Count, x::Number) = Count(seq.data/x, seq.gene, seq.barcode)
%(seq::Count, x::Number) = Count(seq.data%x, seq.gene, seq.barcode)
^(seq::Count, x::Number) = Count(seq.data^x, seq.gene, seq.barcode)

# ------------------------------------------------------------------------
# matrix interface

size(seq::Count) = size(seq.data)

# -- indexing
IndexStyle(seq::Count) = IndexCartesian()

# integer indexes
getindex(seq::Count, I::Vararg{Int,2}) = getindex(seq.data, I...)
getindex(seq::Count, I::AbstractArray{<:Integer}, J) = Count(getindex(seq.data,I,J),seq.gene[I],seq.barcode[J]) 
getindex(seq::Count, I, J::AbstractArray{<:Integer}) = Count(getindex(seq.data,I,J),seq.gene[I],seq.barcode[J]) 
getindex(seq::Count, I::AbstractArray{<:Integer}, J::AbstractArray{<:Integer}) = Count(getindex(seq.data,I,J),seq.gene[I],seq.barcode[J]) 

setindex!(seq::Count, v, I::Vararg{Int,2}) = setindex!(seq.data, v, I...)

# string indexes
# TODO: should we allow for _just_ accessing barcodes?

function getindex(seq::Count, gene::String...)
    length(gene) == 0 && throw(BoundsError(seq, []))

    index = [findfirst(seq.gene .== g) for g in gene]
    getindex(seq, index, :)
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

# ------------------------------------------------------------------------
# creation operators

function scRNAload(dir::AbstractString; batch=missing)
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

# NOTE: will be different merging strategy
#       only common genes (intersection) are kept
function ∩(seq₁::Count{T}, seq₂::Count{S}) where {T <: Real, S <: Real}
    error("need to implement")
end

# ------------------------------------------------------------------------
# filtering

function filtergene(f, seq::Count)
    ι = [f(seq[i,:]) for i in 1:ngenes(seq)]
    return seq[ι,:]
end

function filtercell(f, seq::Count)
    ι = [f(seq[:,i]) for i in 1:ncells(seq)]
    return seq[:,ι]
end

# ------------------------------------------------------------------------
# normalization

function make_loss(x⃗, X₁, X₂, β̄₁, δβ₁¯², β̄₂, δβ₂¯²)
	function f(Θ)
		α, β₁, β₂, γ = Θ
		
		G₁ = (loggamma(x + γ) - loggamma.(x+1) .- loggamma(γ) for x ∈ x⃗)
		G₂ = (x*(α+β₁*x₁+β₂*x₂)-(x+γ)*log(exp(α+β₁*x₁+β₂*x₂) .+ γ) for (x,x₁,x₂) ∈ zip(x⃗,X₁,X₂))

        return -mean(g₁+g₂+γ*log(γ) for (g₁,g₂) ∈ zip(G₁,G₂)) + 0.5*(δβ₁¯²*(β₁-β̄₁)^2 + δβ₂¯²*(β₂-β̄₂)^2)
	end
	
    # NOTE: gradient & hessian have NOT been updated to reflex the batch-specific gradients
	function ∂f!(∂Θ, Θ)
		α, β, γ, σ = Θ
		
		Mu 	 = (exp(+α + β*g) for g ∈ ḡ)
		Mu¯¹ = (exp(-α - β*g) for g ∈ ḡ)

		G₁ = (x-((γ+x)/(1+γ*μ¯¹)) for (x,μ¯¹) ∈ zip(x⃗,Mu¯¹))
		G₂ = (digamma(x+γ)-digamma(γ)+log(γ)+1-log(γ+μ)-(x+γ)/(γ+μ) for (x,μ) ∈ zip(x⃗,Mu))

		∂Θ[1] = -mean(G₁)
        ∂Θ[2] = -mean(g₁*g for (g₁,g) ∈ zip(G₁,ḡ)) + δβ¯²*(β-β₀)
		∂Θ[3] = -mean(G₂)
	end
	
	function ∂²f!(∂²Θ, Θ)
		α, β, γ, σ = Θ
		
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

# TODO: allow for 3 OR 4 parameters depending on if different batches are detected
#       for now we have hardcoded in batch behavior

# input parameters are:
# x  => vector of genes to fit
# ḡ₁ => average gene expression for given cell
# ḡ₂ => average gene expression for given batch
function fit(x, ḡ₁, ḡ₂; β̄₁=1, δβ₁¯²=0, β̄₂=0, δβ₂¯²=0, check_grad=false)
	μ  = mean(x)
	Θ₀ = [
		log(μ),
		β̄₁,
        β̄₂,
		μ^2 / (var(x)-μ),
	]
	
    if Θ₀[end] < 0 || isinf(Θ₀[end])
        Θ₀[end] = 1
	end
	
	loss = check_grad ? TwiceDifferentiable(
		make_loss(x, ḡ₁, ḡ₂, β̄₁, δβ₁¯², β̄₂, δβ₂¯²)[1],
		Θ₀;
		autodiff=:forward
	) : TwiceDifferentiable(
		make_loss(x, ḡ₁, ḡ₂, β̄₁, δβ₁¯², β̄₂, δβ₂¯²)...,
		Θ₀
	)
	
	if check_grad && false
		f, ∂f!, ∂²f! = make_loss(x, ḡ₁, ḡ₂, β̄₁, δβ₁¯²)
		∂, ∂² = zeros(4), zeros(4,4)

		∂f!(∂,Θ₀)
		∂²f!(∂²,Θ₀)

		@show gradient!(loss, Θ₀)
		@show ∂

		@show hessian!(loss, Θ₀)
		@show ∂²
	end

	constraint = TwiceDifferentiableConstraints(
		[-∞, -∞, -∞, 0],
		[+∞, +∞, +∞, +∞],
	)

	soln = optimize(loss, constraint, Θ₀, IPNewton())
	# TODO: check if was successful
	
	Θ̂  = Optim.minimizer(soln)
	Ê  = Optim.minimum(soln)
	δΘ̂ = diag(inv(hessian!(loss, Θ̂)))
	
	# pearson residuals
	α̂, β̂₁, β̂₂, γ̂ = Θ̂
	μ̂ = exp.(α̂ .+ β̂₁*ḡ₁ .+ β̂₂*ḡ₂)
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

# input parameters are:
# β̄₁    => mean of prior on β₁ (cell-specific count)
# δβ₁¯² => uncertainty of prior on β₁ (cell-specific count)
function normalize(seq::Count; β̄₁=1, δβ₁¯²=0, β̄₂=0, δβ₂¯²=0)
    ḡ₁ = vec(mean(seq, dims=1))
    ḡ₂ = zeros(ncells(seq))

    for b ∈ batches(seq)
        ι      = inbatch(seq, b)
        ḡ₂[ι] .= mean(seq[:,ι][:])
    end

    ḡ₁ = log.(ḡ₁)
    ḡ₂ = log.(ḡ₂)

    r = [fit(vec(seq.data[i,:]), ḡ₁, ḡ₂; β̄₁=β̄₁, δβ₁¯²=δβ₁¯², β̄₂=β̄₂, δβ₂¯²=δβ₂¯², check_grad=true) for i ∈ 1:4000]

    return r, vec(mean(seq, dims=2))[1:length(r)]
end

# ------------------------------------------------------------------------
# point of entry for testing

const SEQ1 = "/home/nolln/root/data/seqspace/raw/rep4"
const SEQ2 = "/home/nolln/root/data/seqspace/raw/rep5"
const SEQ3 = "/home/nolln/root/data/seqspace/raw/rep6"

function test()
    seq₁ = scRNAload(SEQ1; batch="rep4")
    seq₂ = scRNAload(SEQ2; batch="rep5")
    seq₃ = scRNAload(SEQ2; batch="rep6")

    seq = seq₁ ∪ seq₂

    @show size(seq)
    seq = filtergene(seq) do gene
        sum(gene .> 0) >= 3            # has to be found in at least 3 cells
    end
    @show size(seq)
    seq = filtercell(seq) do cell
        sum(cell) >= 1e-1*ngenes(seq)  # has to average .1 expression across gens
    end
    @show size(seq)

    return normalize(seq; δβ₁¯²=1e-2, δβ₂¯²=1e-2)
end

end
