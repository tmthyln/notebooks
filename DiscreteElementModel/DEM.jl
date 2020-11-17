### A Pluto.jl notebook ###
# v0.12.3

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

# ╔═╡ 403ea6ce-0202-11eb-2210-39758fd41ad0
using LinearAlgebra, DifferentialEquations

# ╔═╡ f9f12da2-02de-11eb-18bb-5b80c547bdfc
using CUDA

# ╔═╡ 6cdb8850-036c-11eb-1acb-dffe813df288
using BenchmarkTools

# ╔═╡ db757a40-036d-11eb-2d46-efe1cb4a8665
using NearestNeighbors

# ╔═╡ 5eb5f410-0289-11eb-0df0-a76897e82277
using Plots, PlutoUI

# ╔═╡ 07b03400-0266-11eb-1143-e3a94b5bbd3f
md"""
# Discrete Element Modeling for Soil Simulation
This code is derived from the Discrete Element Method described in the paper "Discrete element modelling (DEM) input parameters: understanding their impact on model predictions using statistical analysis" by Yan et al.

## Summary of the Discrete Element Method (Translational Case Only; No Rotation)
The motion of individual particles are determined via typical equations of motion:

$$m_i\dfrac{d\vec{v}_i}{dt} = \sum_j \bigg( \vec{F}_{ij}^n + \vec{F}_ij^t \bigg) + m_i \vec{g}$$

Here, we have
* the mass $m_i$ of particle $i$
* the gravity vector $\vec{g}$
* the translational velocity $\vec{v}_i$ of particle $i$
* the normal force $\vec{F}_{ij}^n$ due to particle $i$ interacting with particle $j$
* the tangential force $\vec{F}_{ij}^t$ due to particle $i$ interacting with particle $j$

Now, we will recursively give the definitions for $\vec{F}_{ij}^n$ and $\vec{F}_{ij}^t$:

$$\vec{F}_{ij}^n = \dfrac{-4}{3}E^*\sqrt{R^*\delta_{ij}^n}\vec{\delta}_{ij}^n - 2\sqrt{\dfrac{5}{6}}\psi\sqrt{C_nm^*}\vec{v}_{ij}^n$$

$$\vec{F}_{ij}^t = -8G^*\sqrt{R^*\delta_{ij}^n}\vec{\delta}_{ij}^t - 2\sqrt{\dfrac{5}{6}}\psi\sqrt{C_t m^*}\vec{v}_{ij}^t$$

where we have
* the equivalent Young's modulus $E^*$: $\dfrac{1}{E^*} = \dfrac{1-v_i^2}{E_i} + \dfrac{1-v_j^2}{E_j}$
* the Poisson's ratio $v_i$
* the equivalent radius $R^*$: $\dfrac{1}{R^*} = \dfrac{1}{R_i} + \dfrac{1}{R_j}$
* the equivalent mass $m^*$: $\dfrac{1}{m^*} = \dfrac{1}{m_i} + \dfrac{1}{m_j}$
* the normal and tangential components of the relative velocity $\vec{v}_{ij}^n$ and $\vec{v}_{ij}^t$ at the contact
* the normal contact overlap $\vec{\delta}_{ij}^n$: $\left\vert \vec{\delta}_{ij}^n \right\vert = R_i + R_j - d_{ij}$, where $d_{ij}$ is the distance of the center of particles
* the tangential contact overlap $\vec{\delta}_{ij}^t$: $\left\vert \vec{\delta}_{ij}^t \right\vert = \int_0^t \left\vert \vec{v}_{ij}^t \right\vert\,dt$ (integral of tangential relative velocity through the collision time)
* the normal contact stiffness $C_n = 2E^*\sqrt{R^*\delta_{ij}^n}$
* the tangential contact stiffness $C_t = 8G^*\sqrt{R^*\delta_{ij}^n}$
* the equivalent shear modulus $G^*$: $\dfrac{1}{G^*} = \dfrac{2(2-v_i)(1+v_i)}{E_i} + \dfrac{2(2-v_j)(1+v_j)}{E_j}$
* the damping ratio coefficient $\psi = \ln(\epsilon)/\sqrt{\ln^2(\epsilon) + \pi^2}$, where $\epsilon$ is the coefficient of restitution

If the relation $\left\vert \vec{F}_{ij}^t \right\vert < \mu_s \left\vert \vec{F}_{ij}^n \right\vert$ is satisfied, then the above equation for $\vec{F}_{ij}^t$ can be used. Otherwise, Coulomb's friction law is needed:

$$\vec{F}_{ij}^t = -\mu_s \left\vert \vec{F}_{ij}^n \right\vert \dfrac{\vec{\delta}_{ij}^t}{\left\vert \vec{\delta}_{ij}^t \right\vert}$$

*Note that for some quantities, having different numbers of (subscript) indices indicates different quantities. For example, $\vec{v}_{ij}^n$ is the normal component of the relative velocity between particles $i$ and $j$ but $\vec{v}_i$ is the velocity vector for particle $i$.*

(See paper for details.)
"""

# ╔═╡ dc8158c0-026c-11eb-179c-bbdd75d272b1
md"""
## Implementation
For a particular point (soil particle), it has the following kinematic/dynamic quantities:
* position in space
* velocity
* (acceleration)
* net force

and has the following properties (roughly constant):
* radius
* mass
* Young's modulus
* Poisson's ratio

Overall, the problem assumes that the dynamics between particles are roughly the same, so all points share
* damping ratio coefficient

Most of the other quantities can be calculated from these ones. First, we'll write some convenience functions to do that:

"""

# ╔═╡ 6c5b2210-02da-11eb-3244-29a66f83337c
begin
	g = [0; -9.81]
	
	R = 1e-3 # m
	R_equiv = R / 2
	mass = 6e-6 # kg
	mass_equiv = mass / 2
	ϵ #= coefficient of restitution =# = 0.45 # acts as a damper during collisions
	ψ #= damping ratio coefficient =# = log.(ϵ) ./ sqrt.(log.(ϵ).^2 + π^2)
	μ_s #= static friction coefficient =# = 0.6
	poissons_ratio = 0.45;
	E #= Young's modulus =# = 0.2e9; # Pa
	E_equiv = E ./ (2 .* (1-poissons_ratio^2));
	G #= shear modulus =# = 0.2e9; # Pa
	G_equiv = E / (4 .* (2 .- poissons_ratio) .* (1 .+ poissons_ratio));
	
	width = 0.15 # m
	depth = -0.05 # m, always negative
	
	parameters = (; g, R, R_equiv, mass, mass_equiv, ϵ, ψ, μ_s, poissons_ratio, E, E_equiv, G, G_equiv, width, depth);
end;

# ╔═╡ 916bf360-01ed-11eb-335b-075c0a87a914
numx = length(R:3*R:width - R);

# ╔═╡ 43aefe20-01f1-11eb-1f1b-ab08cf0fd5f6
numy = length(-R:-3*R:depth + R);

# ╔═╡ 38360e30-02cd-11eb-2722-1f4293594c52
t_R #= critical time step =# = π*10^-3*sqrt(2.6e3/0.2e9)/(0.01631*poissons_ratio + 0.8766)

# ╔═╡ 8ed17cf0-01ee-11eb-39f8-a50075c34f7e
# x-coordinates of particles, initial
x = repeat(R:3*R:width - R, inner=numy)

# ╔═╡ b25aa020-01ee-11eb-0f7c-ad65fb78df07
# y-coordinates of particles, initial
y = repeat(-R:-3*R:depth + R, outer=numx)

# ╔═╡ 00e372a0-0274-11eb-2e03-dbb584c3dea6
num_particles = length(x);

# ╔═╡ 413e8cd0-0211-11eb-044f-bdce00109f46
particles = [
	x' .+ 0.1*R*rand(size(x)); 
	y' .+ 0.1*R*rand(size(x)); 
	zeros(2, num_particles)
]

# ╔═╡ 40f4ed40-035c-11eb-197b-e5992fb5707a
md"""
The following differential equation uses this setup:

We represent our variables of interest (the position and velocity of each point) as a matrix $u$:

$$u = \begin{bmatrix}  \end{bmatrix}$$
"""

# ╔═╡ f4ce67a2-027c-11eb-28d2-edc160f5d146
@inline pos(particles, i) = @view particles[1:2, i]

# ╔═╡ a184e1e0-027d-11eb-253e-d9dd44c1758d
@inline vel(particles, i) = @view particles[3:4, i]

# ╔═╡ 441ef850-0341-11eb-3191-fd41569fdd5b
function particle_particle_interact(du, u, p)
	Cn_coefficient = 2 * p.E_equiv * sqrt(p.R_equiv)
	F_coefficient1 = -4 / 3 * p.E_equiv * sqrt(p.R_equiv)
	F_coefficient2 = -2 * sqrt(5 / 6) * p.ψ * sqrt(p.mass_equiv)
	
	tree = BallTree(u[1:2, :])
	for i in 1:size(u, 2), j in inrange(tree, u[1:2, i], 2*p.R)
		pos_i = pos(u, i); vel_i = vel(u, i); pos_j = pos(u, j); vel_j = vel(u, j)
		
		if i ≠ j
			δn = 2 * p.R - norm(pos_i - pos_j)
			
			rel_vel = vel_i - vel_j
			unit_normal = (pos_j - pos_i) ./ norm(pos_j - pos_i)
			
			vn = dot(rel_vel, unit_normal) .* unit_normal
			δn_vec = δn .* unit_normal
			
			Cn = Cn_coefficient .* sqrt.(δn)
			
			Fnormal = (F_coefficient1 .* sqrt.(δn) .* δn_vec .+ 
				F_coefficient2 .* sqrt.(Cn) .* vn)
			
			du[3:4, i] += Fnormal
		end
	end
end

# ╔═╡ 47c164e0-036c-11eb-1b45-31a10fbd283e
@benchmark particle_particle_interact(zeros(4, test_n * test_n), test_u, parameters)

# ╔═╡ c6e1de00-036a-11eb-3ce6-6190258cb911


# ╔═╡ 94b62820-0368-11eb-2e66-419cac0bb60f


# ╔═╡ 98ee3e42-0341-11eb-3e83-d95c61d4cdef
function particle_barrier_interact(du, u, p)
	Cn_coefficient = 2 .* p.E .* sqrt.(p.R)
	F_coefficient1 = -4 / 3 * p.E / (1 - p.poissons_ratio^2) * sqrt(p.R)
	F_coefficient2 = -2 * sqrt(5/6) * p.ψ * sqrt(p.mass)
	
	for i in 1:size(u, 2)
		δn = p.R - abs(p.depth - pos(u, i)[2])
		
		if δn > 1e-9
			vn = - vel(u, i)[2]
			Cn = Cn_coefficient .* sqrt.(δn)
			Fnormal = (F_coefficient1 .* sqrt.(δn) .* δn + 
				F_coefficient2 .* sqrt.(Cn) .* vn)
			du[4, i] -= Fnormal
		end
	end
end

# ╔═╡ 0b6760c0-0b47-11eb-0e5a-d1594c0c9ad7


# ╔═╡ d011998e-0b47-11eb-32b4-211030036737


# ╔═╡ 09563322-0277-11eb-21ac-2de8b7564603
function dem!(du, u, p, t)
	du[1:2, :] .= u[3:4, :]
	du[3:4, :] .= 0
	
	# particle-particle interaction forces
	particle_particle_interact(du, u, p)
	
	# particle-floor/wall interaction forces
	particle_barrier_interact(du, u, p)
	
	# divide by mass to get acceleration and add acceleration due to gravity
	du[3:4, :] .= du[3:4, :] ./ p.mass .+ p.g
	
	return nothing
end

# ╔═╡ 166af9e0-0279-11eb-07b7-4d70fabcff8d
tspan = (0.0, 0.2)

# ╔═╡ 2c7f9830-0279-11eb-1f9a-0b34ee0323ff
prob = ODEProblem(dem!, particles, tspan, parameters)

# ╔═╡ 44061600-0279-11eb-0775-55a3d90bcce5
sol = solve(prob, dt = 0.1*t_R, adaptive=false)#, alg_hints=[:stiff])

# ╔═╡ d8d3fcf0-0384-11eb-2d3e-839d79b3f139
length(sol)

# ╔═╡ 0be3fd60-028c-11eb-1faa-c59140de270b
function circle(h, k, r)
	Θ = LinRange(0, 2*π, 500)
	h .+ r .* sin.(Θ), k .+ r.*cos.(Θ)
end

# ╔═╡ 25eed1f0-02d5-11eb-3401-c78abceaa4ee
function soilplot(h, k, r=R)
	plot(circle.(h, k, r), 
		#xlims = [0, width], ylims=[2*depth, 0],
		seriestype=[:shape], legend=false, aspectratio=1)
end

# ╔═╡ d2c5f9c0-02b8-11eb-3ac8-c99e148fd2c3
md"Time Step of Solution: $(@bind timestep Slider(1:(length(sol)÷500):length(sol), show_value=true))"

# ╔═╡ bd331290-028c-11eb-1fda-99229a2bdfbe
soilplot(sol[timestep][1, :], sol[timestep][2, :])

# ╔═╡ a63c4770-02d5-11eb-387f-3527351c6433
md"""
@gif for i in eachindex(sol)
	soilplot(sol[i][1, :], sol[i][2, :])
end every 750
"""

# ╔═╡ Cell order:
# ╟─07b03400-0266-11eb-1143-e3a94b5bbd3f
# ╟─dc8158c0-026c-11eb-179c-bbdd75d272b1
# ╠═403ea6ce-0202-11eb-2210-39758fd41ad0
# ╠═f9f12da2-02de-11eb-18bb-5b80c547bdfc
# ╠═6c5b2210-02da-11eb-3244-29a66f83337c
# ╠═916bf360-01ed-11eb-335b-075c0a87a914
# ╠═43aefe20-01f1-11eb-1f1b-ab08cf0fd5f6
# ╠═38360e30-02cd-11eb-2722-1f4293594c52
# ╠═8ed17cf0-01ee-11eb-39f8-a50075c34f7e
# ╠═b25aa020-01ee-11eb-0f7c-ad65fb78df07
# ╠═00e372a0-0274-11eb-2e03-dbb584c3dea6
# ╠═413e8cd0-0211-11eb-044f-bdce00109f46
# ╠═40f4ed40-035c-11eb-197b-e5992fb5707a
# ╠═f4ce67a2-027c-11eb-28d2-edc160f5d146
# ╠═a184e1e0-027d-11eb-253e-d9dd44c1758d
# ╠═441ef850-0341-11eb-3191-fd41569fdd5b
# ╠═47c164e0-036c-11eb-1b45-31a10fbd283e
# ╠═6cdb8850-036c-11eb-1acb-dffe813df288
# ╠═db757a40-036d-11eb-2d46-efe1cb4a8665
# ╟─c6e1de00-036a-11eb-3ce6-6190258cb911
# ╟─94b62820-0368-11eb-2e66-419cac0bb60f
# ╠═98ee3e42-0341-11eb-3e83-d95c61d4cdef
# ╠═0b6760c0-0b47-11eb-0e5a-d1594c0c9ad7
# ╠═d011998e-0b47-11eb-32b4-211030036737
# ╠═09563322-0277-11eb-21ac-2de8b7564603
# ╠═166af9e0-0279-11eb-07b7-4d70fabcff8d
# ╠═2c7f9830-0279-11eb-1f9a-0b34ee0323ff
# ╠═44061600-0279-11eb-0775-55a3d90bcce5
# ╠═5eb5f410-0289-11eb-0df0-a76897e82277
# ╠═d8d3fcf0-0384-11eb-2d3e-839d79b3f139
# ╠═0be3fd60-028c-11eb-1faa-c59140de270b
# ╠═25eed1f0-02d5-11eb-3401-c78abceaa4ee
# ╠═d2c5f9c0-02b8-11eb-3ac8-c99e148fd2c3
# ╟─bd331290-028c-11eb-1fda-99229a2bdfbe
# ╠═a63c4770-02d5-11eb-387f-3527351c6433
