using SparseArrays
using FillArrays

include("space_dimensions.jl")
include("even_odd_classification.jl")
include("staggered_grid.jl")
include("solver_variable.jl")
include("solver_core.jl")
include("boundary_conditions.jl")
include("solver.jl")

M = (50, 1, 1)
N = 3
N_steps = 3
xdim = (0.0, 1.0)
# ydim = DTYPE.(convert_strip_length.((-500.0u"nm", 500.0u"nm")))
ydim = 0.
# zdim = DTYPE.(convert_strip_length.((-500.0u"nm", 500.0u"nm")))
zdim = 0.
gspec = make_gridspec(M, xdim, ydim, zdim)

S = StaggeredGridVariable{Fill{Float64, 1}}([3, 3, 3, 3, 3, 3, 3, 3], gspec)

dump(S)



StaggeredGridVariable{Float64}(gspec)

StaggeredGridVariable{Vector{Float64}}(3, gspec)
StaggeredGridVariable{Vector{Float64}}([1, 2, 3, 4, 5, 6, 7, 8], gspec)
StaggeredGridVariable{Fill{T, 1}} =
StaggeredGridVariable{Fill{Float64, 1}}(3, gspec)

u = SolverVariable{Float64}(3, 10, (0., 1.), (0., 1.), (0., 1.))
function fill_func(x, i)
    if x[2] < 0.5
        return [0.0, 1.0, 2.0][i]
    else
        return[0.5, 0.5, 0.5][i]
    end
end

fill_array!(u, fill_func)

u
