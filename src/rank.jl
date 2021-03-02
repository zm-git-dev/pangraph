module SoftRank

import ChainRulesCore: rrule, NO_FIELDS

export rank, softrank

function partition(x; ϵ=1e-9)
	if length(x) == 0
		return eltype(x)[]
	end
	
	sizes = [1]
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
	w = vec(collect(n:-1:1))
	ι = reverse(sortperm(x))
	s = x[ι]

	dual   = isotonic(s.-w)
	primal = s .- dual
	
	return primal[invperm(ι)]
end

rank(x)             = invperm(sortperm(x))
softrank(x; ϵ=1e-3) = projection(x ./ ϵ)
	
function rrule(::typeof(softrank), x; ϵ=1e-3)
	# scale input
	x = x ./ ϵ
	
	# compute value
	n = length(x)
	w = vec(collect(n:-1:1))
	ι = reverse(sortperm(x))
	s = x[ι]

	dual   = isotonic(s.-w)
	primal = s .- dual
	
	ι¯¹ = invperm(ι)	
	return primal[ι¯¹], (∇) -> (NO_FIELDS, ∇ .- ∇isotonic(dual, ∇)[ι¯¹])
end

end
