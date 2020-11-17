### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 5b4b8f10-ff4b-11ea-03bc-b7b57b951a91
begin
	import Pkg; Pkg.activate(".");
	using Primes
	using Plots
	using PlutoUI
end

# ╔═╡ d83ed9e0-ff47-11ea-159f-47bb0701ed5a
md"""
# Turn Left at the Next Prime
Notebook inspired by a [blog post by John Cook](https://www.johndcook.com/blog/2020/09/24/gaussian_integer_walk/) to make it interactive and improve performance.

The basic premise is this: start at some point in the complex plane (with integer coefficients), facing east. Move unit-by-unit until the number is a Gaussian prime. If Gaussian prime, turn left.

First, a quick function to determine whether a complex integer is a Gaussian prime.
"""

# ╔═╡ fb7553be-ff48-11ea-1ff3-65e2f9cadf04
function isgaussprime(z::Complex{<:Integer})
    a, b = real(z), imag(z)
    if a * b != 0
        return isprime(a^2 + b^2)
    else
        c = abs(a+b)
        return isprime(c) && c % 4 == 3
	end
end

# ╔═╡ 399da040-ff66-11ea-2740-ab721778850e
md"""
Now, to make the interactivity a bit cleaner, I'm going to split the functionality of the walk into 3 functions:
* `walklimit`: number of iterations to return to the same point
* `walkbounds`: the bounds of the space that the values take on
* `walk`: the set of points walked by the algorithm

The functions `walklimit` and `walkbounds` are just to fix the range of certain interactive visual elements.
"""

# ╔═╡ f5ed8060-ff54-11ea-0753-29c38bebd06e
function walklimit(start)
	limit = 1
	z = start
	delta = 1
	
	while true
		z = z + delta
		
		isgaussprime(z) && (delta *= im)
		z == start && break
		limit += 1
	end
	
	limit + 1
end

# ╔═╡ acbf25de-ff56-11ea-1ad4-b7a50f0f8cdb
function walkbounds(start)
	min_real, max_real, min_imag, max_imag = real(start), real(start), imag(start), imag(start)
	z = start
	delta = 1
	
	while true
		z = z + delta
		
		min_real = min(min_real, real(z))
		max_real = max(max_real, real(z))
		min_imag = min(min_imag, imag(z))
		max_imag = max(max_imag, imag(z))
		
		isgaussprime(z) && (delta *= im)
		z == start && break
	end
	
	(min_real, max_real), (min_imag, max_imag)
end

# ╔═╡ 28ef0530-ff49-11ea-37cf-313928410e83
function walk(start, limit = 10)
	points = [start]
	z = start
	delta = 1
	
	while limit > 0
		z = z + delta
		push!(points, z)
		
		isgaussprime(z) && (delta *= im)
		z == start && break
		limit -= 1
	end
	
	points
end

# ╔═╡ 08d1f7ee-ff56-11ea-3c47-87c30f5f066b
md"Real Part: $(@bind realP Slider(0:200, default=3)) Imaginary Part: $(@bind imagP Slider(0:200, default=5))"

# ╔═╡ 24e9bdb0-ff56-11ea-0c99-7b3f6712252b
begin
	point = complex(realP, imagP)
	md"""
	## Gaussian Prime Spiral
	Starting at the point $realP + $(imagP)i.
	"""
end

# ╔═╡ f2ad5b70-ff52-11ea-2e57-1be6b16a23da
md"$(@bind enforce_limit CheckBox()) Limit: $(@bind limit Slider(1:walklimit(point), show_value=true))"

# ╔═╡ 2d061640-ff58-11ea-23a8-fb4942f21081
bounds = walkbounds(point);

# ╔═╡ 60544cc2-ff52-11ea-00f6-a727cfe14a89
plot(
	walk(point, (enforce_limit ? limit : 10000000)), 
	xlims=bounds[1] .+ [-1, 1], 
	ylims=bounds[2] .+ [-1, 1], 
	legend=false)

# ╔═╡ dc628560-ff67-11ea-3f81-d933036c21ca
function walkmesh(xrange, yrange, threshold)
	points = Complex{Int}[]
	Threads.@threads for x = xrange
		for y = yrange
			l = walklimit(complex(x, y))
			if l > threshold
				push!(points, complex(x, y))
			end
		end
	end
	points
end

# ╔═╡ 43484480-ff69-11ea-3f30-6fff96055bd9
@bind threshold Slider(1:1500, default = 5, show_value = true)

# ╔═╡ be316010-ff68-11ea-3f26-7dd4d555c24a
scatter(walkmesh(-30:30, -30:30, threshold), legend=false)

# ╔═╡ 06c7ae90-ff8e-11ea-051a-6bfabeecd061
md"""
There are some "rapid" changes in the number of points that meet a threshold. For example, see
* 412-413
* 1316-1317
"""

# ╔═╡ Cell order:
# ╠═5b4b8f10-ff4b-11ea-03bc-b7b57b951a91
# ╟─d83ed9e0-ff47-11ea-159f-47bb0701ed5a
# ╠═fb7553be-ff48-11ea-1ff3-65e2f9cadf04
# ╟─399da040-ff66-11ea-2740-ab721778850e
# ╠═f5ed8060-ff54-11ea-0753-29c38bebd06e
# ╠═acbf25de-ff56-11ea-1ad4-b7a50f0f8cdb
# ╠═28ef0530-ff49-11ea-37cf-313928410e83
# ╟─24e9bdb0-ff56-11ea-0c99-7b3f6712252b
# ╟─08d1f7ee-ff56-11ea-3c47-87c30f5f066b
# ╟─f2ad5b70-ff52-11ea-2e57-1be6b16a23da
# ╟─2d061640-ff58-11ea-23a8-fb4942f21081
# ╟─60544cc2-ff52-11ea-00f6-a727cfe14a89
# ╠═dc628560-ff67-11ea-3f81-d933036c21ca
# ╠═43484480-ff69-11ea-3f30-6fff96055bd9
# ╠═be316010-ff68-11ea-3f26-7dd4d555c24a
# ╠═06c7ae90-ff8e-11ea-051a-6bfabeecd061
