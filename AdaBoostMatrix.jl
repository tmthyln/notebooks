### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ decaea40-13e9-11eb-1f00-67b8afa1cd43
using LinearAlgebra

# ╔═╡ 63978fe0-1556-11eb-1c80-4b62e5591ffa
md"""
# AdaBoost (CMSC828U HW4)
"""

# ╔═╡ cf94e040-13de-11eb-38a1-f713eda5ba34
M = [
	-1 1  1  1  1 -1 -1  1;
	-1 1  1 -1 -1  1  1  1;
	 1 -1 1  1  1 -1  1  1;
	 1 -1 1  1 -1  1  1  1;
	 1 -1 1 -1  1  1  1 -1;
	 1 1 -1  1  1  1  1 -1;
	 1 1 -1  1  1  1 -1  1;
	 1 1  1  1 -1 -1  1 -1
	]

# ╔═╡ 61b27720-13e0-11eb-2ce1-c58dad8f696b
tmax = 7; (m, n) = size(M);

# ╔═╡ 7a388a9e-13e0-11eb-3b29-15874a1cdaec
d₁ = [(3 - sqrt(5)) / 8, (3 - sqrt(5)) / 8, 1/6, 1/6, 1/6, (sqrt(5) - 1) / 8, (sqrt(5) - 1) / 8, 0]

# ╔═╡ c4a3dfb0-13e3-11eb-24e0-05dd36532497
function adaboost(M, tmax, d₁)
	λ = zeros(size(M, 2))
		
	for t in 1:tmax
		if t == 1
			d = d₁
		else
			e = exp.(-M * λ)
			d = e ./ sum(e, dims=1)
		end
		j = argmax(d' * M)
		r = (d' * M)[j]
		α = 0.5 * log((1+r)/(1-r))
		λ[j[2]] += α
		println("Choosing model $(j[2])")
	end
	λ / norm(λ, 1)
end

# ╔═╡ 5de1a9f0-13e9-11eb-136f-41ce2ce8b473
adaboost(M, tmax, d₁)

# ╔═╡ 90fab050-148b-11eb-1cb1-f76745ed49b2
margin(M, α) = minimum(M * α)

# ╔═╡ 27aa35f0-148e-11eb-121e-39d99e6c2f41
margin(M, adaboost(M, tmax, d₁))

# ╔═╡ 4236a0c0-148e-11eb-315b-377b705487c6
margin(M, [2, 3, 4, 1, 2, 2, 1, 1] ./ 16)

# ╔═╡ de4cadc0-1492-11eb-0c93-6bf2f2841015


# ╔═╡ Cell order:
# ╟─63978fe0-1556-11eb-1c80-4b62e5591ffa
# ╠═decaea40-13e9-11eb-1f00-67b8afa1cd43
# ╟─cf94e040-13de-11eb-38a1-f713eda5ba34
# ╠═61b27720-13e0-11eb-2ce1-c58dad8f696b
# ╠═7a388a9e-13e0-11eb-3b29-15874a1cdaec
# ╠═c4a3dfb0-13e3-11eb-24e0-05dd36532497
# ╠═5de1a9f0-13e9-11eb-136f-41ce2ce8b473
# ╠═90fab050-148b-11eb-1cb1-f76745ed49b2
# ╠═27aa35f0-148e-11eb-121e-39d99e6c2f41
# ╠═4236a0c0-148e-11eb-315b-377b705487c6
# ╠═de4cadc0-1492-11eb-0c93-6bf2f2841015
