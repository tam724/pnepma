##
include("../src/StaRMAP.jl")
using .StaRMAP
using SparseArrays
using Distributions
using LinearAlgebra
using StaticArrays
using LuxurySparse

##

M = (50, 50, 50)
gspec = make_gridspec(M, (0., 1.), (0., 1.), (0., 1.))

u = zeros(StaRMAP.MStaggeredGridVariable{1, 1, 1, 0, 1, 0, 0, 0, Float64}, gspec)

b = zeros(StaRMAP.MBoundaryGridVariableX{1, 1, 1, 1, Float64}, gspec)
##
Ax = StaRMAP.SAdvectionMatrixX{1, 1, 1, 0, 1, 0, 0, 0, Float64}(
    ones(1, 1),       #eee
    zeros(1, 0),      #eeo
    zeros(1, 0),      #eoe
    zeros(0, 0),      #eoo
    ones(1, 1),       #oee
    zeros(0, 1),      #oeo
    zeros(0, 1),      #ooe
    zeros(0, 0)       #ooo
)
Ay = StaRMAP.SAdvectionMatrixY{1, 1, 1, 0, 1, 0, 0, 0, Float64}(
    ones(1, 1),
    zeros(1, 0),
    ones(1, 1),
    zeros(0, 1),
    zeros(1, 0),
    zeros(0, 0),
    zeros(0, 1),
    zeros(0, 0)
)
Az = StaRMAP.SAdvectionMatrixZ{1, 1, 1, 0, 1, 0, 0, 0, Float64}(
    ones(1, 1),
    ones(1, 1),
    zeros(1, 0),
    zeros(0, 1),
    zeros(1, 0),
    zeros(0, 1),
    zeros(0, 0),
    zeros(0, 0)
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


Bx = BoundaryConditionSource(
    StaRMAP.SBoundaryMatrixX{1, 1, 1, 0, 1, 0, 0, 0, Float64}(spzeros(1, 1), spzeros(0, 1), spzeros(0, 1), spzeros(0, 0)),
    zeros(StaRMAP.MBoundaryGridVariableX{1, 0, 0, 0, Float64}, gspec)
)

By = BoundaryConditionZero(
    StaRMAP.SBoundaryMatrixY{1, 1, 1, 0, 1, 0, 0, 0, Float64}(spzeros(1, 1), spzeros(0, 1), spzeros(0, 1), spzeros(0, 0))
)

Bz = BoundaryConditionZero(
    StaRMAP.SBoundaryMatrixZ{1, 1, 1, 0, 1, 0, 0, 0, Float64}(spzeros(1, 1), spzeros(0, 1), spzeros(0, 1), spzeros(0, 0))
)

S = ones(StaRMAP.MStaggeredGridVariable{1, 1, 1, 0, 1, 0, 0, 0, Float64}, gspec)
Q = ones(StaRMAP.MStaggeredGridVariable{1, 1, 1, 0, 1, 0, 0, 0, Float64}, gspec)
C = ones(StaRMAP.MStaggeredGridVariable{1, 1, 1, 0, 1, 0, 0, 0, Float64}, gspec)

for i in eachindex(Bx.g.oee)
    Bx.g.oee[i] .= 1.0
end
##

initial_func(x) = [pdf(MultivariateNormal([0.5, 0.5, 0.5], Diagonal([0.01, 0.01, 0.01])), collect(x))]
problem = DefaultStaRMAPProblem(Ax, Ay, Az, Bx, By, Bz, S, C, Q, gspec)

u = zeros(StaRMAP.MStaggeredGridVariable{1, 1, 1, 0, 1, 0, 0, 0, Float64}, gspec)
du = zeros(StaRMAP.MStaggeredGridVariable{1, 1, 1, 0, 1, 0, 0, 0, Float64}, gspec)

eee = EvenOddClassification{Even, Even, Even}
# u[eee] .= initial_func.(points(eee, gspec))
##
using Plots

t = 0.
@gif for i in 1:50
    t = step_pde!(problem, u, du, t, 0.005)
    # plot(getindex.(u[eee], 1)[:, :, 1], st=:surface)
    plot(getindex.(u[eee], 1))
end every 1

## 
using BenchmarkTools
using Profile
using PProf
##

@time for i in 1:50
    t = step_pde!(problem, u, du, t, 0.005)
end

##
Profile.clear()

