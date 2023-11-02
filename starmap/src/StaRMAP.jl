module StaRMAP

using SparseArrays
using FillArrays
using StaticArrays
using LinearAlgebra

include("space_dimensions.jl")
include("even_odd_classification.jl")
include("staggered_grid.jl")
include("solver_variable.jl")
include("solver_core.jl")
include("boundary_conditions.jl")
include("solver.jl")

export Dimension, X, Y, Z
export GridSpec, points, make_gridspec, dx, dy, dz, dvol
export EvenOdd, Even, Odd, EvenOddClassification, EvenOddPairity, EvenParitiy, OddPairity
export switch_eo, eo_in, is_even_in, is_odd_in, all_eos, all_eos_eo_in, all_eos_odd_in, all_eos_even_in
export SolverVariable, fill_scalar!, fill_array!
export StaggeredGridVariable, MStaggeredGridVariable
export BoundaryGridVariable, BoundaryGridVariableX, BoundaryGridVariableY, BoundaryGridVariableZ, MBoundaryGridVariableY, MBoundaryGridVariableX, MBoundaryGridVariableZ
export AdvectionMatrix, SAdvectionMatrixX, SAdvectionMatrixY, SAdvectionMatrixZ
export BoundaryMatrix, SBoundaryMatrixX, SBoundaryMatrixY, SBoundaryMatrixZ
export BoundaryCondition, BoundaryConditionSource, BoundaryConditionZero
export StaRMAPProblem, DefaultStaRMAPProblem, step_pde!

end # module StaRMAP
