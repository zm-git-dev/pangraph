module SoftRank

import ChainRulesCore: rrule, NO_FIELDS

export rank, softrank

function partition(x; ϵ=1e-9)
	if length(x) == 0
		return eltype(x)[]
	end
	
	sizes = Int[1]
	for i in 2:length(x)
		if abs(x[i] - x[i-1]) > ϵ
			append!(sizes, 0)
		end
		sizes[end] += 1
	end
	
	return sizes
end

function isotonic(x)
	n = length(x)
	τ = collect(vec(1:n))
	c = ones(n)
	Σ = copy(x)
	y = copy(x)
	
	i = 1
	while i <= n
		k = τ[i] + 1
		if k == (n + 1)
			break
		end
		
		if y[i] > y[k]
			i = k
			continue
		end
		
		Σʸ = Σ[i]
		Σᶜ = c[i]
		# We are within an increasing subsequence
		while true
			y₀  = y[k]
			Σʸ += Σ[k]
			Σᶜ += c[k]
			
			k = τ[k] + 1
			if k == (n + 1) || y₀ > y[k]
				y[i] = Σʸ / Σᶜ
				Σ[i] = Σʸ
				c[i] = Σᶜ
				
				τ[i]   = k-1
				τ[k-1] = i
				if i > 1
					i = τ[i-1]
				end
				
				break
			end
		end
	end
	
	i = 1
	while i <= n
		k = τ[i]
		y[(i+1):k] .= y[i]
		i = k + 1
	end
	
	return y
	
end

function ∇isotonic(soln, x⃗)
	y⃗ = zeros(size(x⃗))
	
	i₁ = 1
	for δ in partition(soln)
		i₂ = i₁ + δ
		y⃗[i₁:(i₂-1)] .= sum(x⃗[i₁:(i₂-1)])/δ
		i₁ = i₂
	end
	
	return y⃗
end

function projection(x)
	n = length(x)
    w = reverse(range(1/n, 1; length=n)) #vec(collect(n:-1:1))
	ι = reverse(sortperm(x))
	s = x[ι]

	dual   = isotonic(s.-w)
	primal = s .- dual
	
	return primal[invperm(ι)]
end

rank(x)             = invperm(sortperm(x))
softrank(x; ϵ=1e-2) = projection(x ./ ϵ)
	
function rrule(::typeof(softrank), x; ϵ=1e-2)
	# scale input
	x = x ./ ϵ
	
	# compute value
	n = length(x)
	w = reverse(range(1/n, 1; length=n))
	ι = reverse(sortperm(x))
	s = x[ι]

	dual   = isotonic(s.-w)
	primal = s .- dual
	
	ι¯¹ = invperm(ι)	
    return primal[ι¯¹], (∇) -> (NO_FIELDS, (∇ .- ∇isotonic(dual, ∇[ι])[ι¯¹]) ./ ϵ)
end

# FIXME: topological flips are wrong!!

function hsplit!(r, i, δx, δy, ι, n)
    x₀ = (δx.hi + δx.lo) / 2

    lo = findall(r[1,:] .≤ x₀)
    hi = findall(r[1,:] .> x₀)

    @show lo, hi

    # left partition
    if length(lo) > 1
        n = vsplit!(r[:,lo], i[lo], (lo=δx.lo,hi=x₀), δy, ι, n)
    elseif length(lo) == 1
        ι[i[lo[1]]] = n
        n += 1
    end

    # right partition
    if length(hi) > 1
        n = vsplit!(r[:,hi], i[hi], (lo=x₀,hi=δx.hi), δy, ι, n)
    elseif length(hi) == 1
        ι[i[hi[1]]] = n
        n += 1
    end

    return n
end

function vsplit!(r, i, δx, δy, ι, n)
    y₀ = (δy.hi + δy.lo) / 2

    lo = findall(r[2,:] .≤ y₀)
    hi = findall(r[2,:] .> y₀)

    @show lo, hi

    # left partition
    if length(lo) > 1
        n = hsplit!(r[:,lo], i[lo], δx, (lo=δy.lo,hi=y₀), ι, n)
    elseif length(lo) == 1
        ι[i[lo[1]]] = n
        n += 1
    end

    # right partition
    if length(hi) > 1
        n = hsplit!(r[:,hi], i[hi], δx, (lo=y₀,hi=δy.hi), ι, n)
    elseif length(hi) == 1
        ι[i[hi[1]]] = n
        n += 1
    end

    return n
end

# NOTE: assumes 2D
#       assumes coordinates fall between -1 and +1
function hilbert(r)
    ι = zeros(Int, size(r,2))
    i = collect(1:size(r,2))
    n = hsplit!(r, i, (lo=-1.0,hi=+1.0), (lo=-1.0,hi=+1.0), ι, 1)

    return ι
end

end
