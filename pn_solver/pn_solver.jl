# module PNSolver

using Zygote
using Zygote: @ignore, @adjoint
using FiniteDifferences
using ForwardDiff

using Flux
using NNlib

using LinearAlgebra
using Distributions
using Combinatorics

using NeXLCore
using Plots
using DataStructures
using FillArrays
using StaticArrays
using SparseArrays
using SpecialFunctions
using SphericalHarmonics
using HCubature
using PeriodicTable
using PhysicalConstants
using Interpolations
using Unitful

include("../starmap/src/StaRMAP.jl")

using .StaRMAP
import .StaRMAP: grid, advection_matrix, boundary_condition, capacity_coefficient, homogeneous_term, source_term

include("units.jl")

include("moment_specification.jl")
include("pn_variable.jl")
include("pn_problem.jl")
include("pn_matrices.jl")
include("pn_parametrizations/pn_material.jl")

include("stopping_power.jl")
include("transport_coefficient.jl")

include("x_ray_intensities.jl")
include("iterators_extensions.jl")
include("pn_intensities.jl")

# export make_pn_solver_variable

# end # module PNSolver
