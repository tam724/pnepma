##
include("../src/StaRMAP.jl")
using .StaRMAP
using SparseArrays
using Distributions
using LinearAlgebra

##
M = (40, 40, 40)
gspec = make_gridspec(M, (0., 1.), (0., 1.), (0., 1.))

var_dims = [1, 1, 1, 0, 1, 0, 0, 0]
u = StaRMAP.StaticStaggeredGridVariable{1, 1, 1, 0, 1, 0, 0, 0, Float64}(gspec)

##
# u = StaggeredGridVariable{Vector{Float64}}(var_dims, gspec)

mat = spzeros(1, 1)
mat[1, 1] = 1.
Ax = AdvectionMatrix{SparseMatrixCSC{Float64}}(
    mat,                #eee
    spzeros(1, 0),      #eeo
    spzeros(1, 0),      #eoe
    spzeros(0, 0),      #eoo
    mat,                #oee
    spzeros(0, 1),      #oeo
    spzeros(0, 1),      #ooe
    spzeros(0, 0)       #ooo
)
Ay = AdvectionMatrix{SparseMatrixCSC{Float64}}(
    mat,
    spzeros(1, 0),
    mat,
    spzeros(0, 1),
    spzeros(1, 0),
    spzeros(0, 0),
    spzeros(0, 1),
    spzeros(0, 0)
)
Az = AdvectionMatrix{SparseMatrixCSC{Float64}}(
    mat,
    mat,
    spzeros(1, 0),
    spzeros(0, 1),
    spzeros(1, 0),
    spzeros(0, 1),
    spzeros(0, 0),
    spzeros(0, 0)
)
#
# eee: 1 ρ
# eeo: 1 w
# eoe: 1 v
# eoo: 0
# oee: 1 u
# oeo: 0
# ooe: 0
# ooo: 0


Bx = BoundaryConditionZero(
    BoundaryMatrixX{Float64}(spzeros(1, 1), spzeros(0, 1), spzeros(0, 1), spzeros(0, 0))
)

By = BoundaryConditionZero(
    BoundaryMatrixY{Float64}(spzeros(1, 1), spzeros(0, 1), spzeros(0, 1), spzeros(0, 0))
)

Bz = BoundaryConditionZero(
    BoundaryMatrixZ{Float64}(spzeros(1, 1), spzeros(0, 1), spzeros(0, 1), spzeros(0, 0))
)

S = StaggeredGridVariable{Vector{Float64}}(var_dims, gspec)
Q = StaggeredGridVariable{Vector{Float64}}(var_dims, gspec)
C = StaggeredGridVariable{Vector{Float64}}(var_dims, gspec)

for eo in all_eos()
    for i in eachindex(S[eo])
        S[eo][i][:] .= 1.
    end
end

##

initial_func(x) = [pdf(MultivariateNormal([0.5, 0.5, 0.5], Diagonal([0.01, 0.01, 0.01])), collect(x))]
problem = DefaultStaRMAPProblem(Ax, Ay, Az, Bx, By, Bz, S, C, Q, gspec)

u = StaggeredGridVariable{Vector{Float64}}(var_dims, gspec)
du = StaggeredGridVariable{Vector{Float64}}(var_dims, gspec)

eee = EvenOddClassification{Even, Even, Even}
u[eee] .= initial_func.(points(eee, gspec))

using Plots
plot(getindex.(u[eee], 1))

t = 0.
@gif for i in 1:100
    t = step_pde!(problem, u, du, t, 0.01)
    plot(getindex.(u[eee], 1))
end every 1

## 
using BenchmarkTools
using Profile

##

@profile for i in 1:100
    t = step_pde!(problem, u, du, t, 0.01)
end

##
Profile.clear()

