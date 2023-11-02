const AdvectionMatrix{T} = EvenOddProperty{Matrix{T}, Matrix{T}, Matrix{T}, Matrix{T}, Matrix{T}, Matrix{T}, Matrix{T}, Matrix{T}}
const AdvectionMatrixCSC{T, Ti} = EvenOddProperty{SparseMatrixCSC{T, Ti}, SparseMatrixCSC{T, Ti}, SparseMatrixCSC{T, Ti}, SparseMatrixCSC{T, Ti}, SparseMatrixCSC{T, Ti}, SparseMatrixCSC{T, Ti}, SparseMatrixCSC{T, Ti}, SparseMatrixCSC{T, Ti}}
const SAdvectionMatrixX{NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T} = EvenOddProperty{SMatrix{NEEE, NOEE, T}, SMatrix{NEEO, NOEO, T}, SMatrix{NEOE, NOOE, T}, SMatrix{NEOO, NOOO, T}, SMatrix{NOEE, NEEE, T}, SMatrix{NOEO, NEEO, T}, SMatrix{NOOE, NEOE, T}, SMatrix{NOOO, NEOO, T}}
const SAdvectionMatrixY{NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T} = EvenOddProperty{SMatrix{NEEE, NEOE, T}, SMatrix{NEEO, NEOO, T}, SMatrix{NEOE, NEEE, T}, SMatrix{NEOO, NEEO, T}, SMatrix{NOEE, NOOE, T}, SMatrix{NOEO, NOOO, T}, SMatrix{NOOE, NOEE, T}, SMatrix{NOOO, NOEO, T}}
const SAdvectionMatrixZ{NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T} = EvenOddProperty{SMatrix{NEEE, NEEO, T}, SMatrix{NEEO, NEEE, T}, SMatrix{NEOE, NEOO, T}, SMatrix{NEOO, NEOE, T}, SMatrix{NOEE, NOEO, T}, SMatrix{NOEO, NOEE, T}, SMatrix{NOOE, NOOO, T}, SMatrix{NOOO, NOOE, T}}

const BoundaryMatrix = EvenOddProperty
const BoundaryMatrixX{T} = EvenOddProperty{Nothing, Nothing, Nothing, Nothing, Matrix{T}, Matrix{T}, Matrix{T}, Matrix{T}}
const BoundaryMatrixY{T} = EvenOddProperty{Nothing, Nothing, Matrix{T}, Matrix{T}, Nothing, Nothing, Matrix{T}, Matrix{T}}
const BoundaryMatrixZ{T} = EvenOddProperty{Nothing, Matrix{T}, Nothing, Matrix{T}, Nothing, Matrix{T}, Nothing, Matrix{T}}

BoundaryMatrixX{T}(Boee, Boeo, Booe, Booo) where {T} = BoundaryMatrixX{T}(nothing, nothing, nothing, nothing, Boee, Boeo, Booe, Booo)
BoundaryMatrixY{T}(Beoe, Beoo, Booe, Booo) where {T} = BoundaryMatrixY{T}(nothing, nothing, Beoe, Beoo, nothing, nothing, Booe, Booo)
BoundaryMatrixZ{T}(Beeo, Beoo, Boeo, Booo) where {T} = BoundaryMatrixZ{T}(nothing, Beeo, nothing, Beoo, nothing, Boeo, nothing, Booo)

const SBoundaryMatrixX{NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T} = EvenOddProperty{Nothing, Nothing, Nothing, Nothing, SMatrix{NOEE, NEEE, T}, SMatrix{NOEO, NEEO, T}, SMatrix{NOOE, NEOE, T}, SMatrix{NOOO, NEOO, T}}
const SBoundaryMatrixY{NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T} = EvenOddProperty{Nothing, Nothing, SMatrix{NEOE, NEEE, T}, SMatrix{NEOO, NEEO, T}, Nothing, Nothing, SMatrix{NOOE, NOEE, T}, SMatrix{NOOO, NOEO, T}}
const SBoundaryMatrixZ{NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T} = EvenOddProperty{Nothing, SMatrix{NEEO, NEEE, T}, Nothing, SMatrix{NEOO, NEOE, T}, Nothing, SMatrix{NOEO, NOEE, T}, Nothing, SMatrix{NOOO, NOOE, T}}

SBoundaryMatrixX{NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T}(Boee, Boeo, Booe, Booo) where {NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T} = SBoundaryMatrixX{NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T}(nothing, nothing, nothing, nothing, Boee, Boeo, Booe, Booo)
SBoundaryMatrixY{NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T}(Beoe, Beoo, Booe, Booo) where {NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T} = SBoundaryMatrixY{NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T}(nothing, nothing, Beoe, Beoo, nothing, nothing, Booe, Booo)
SBoundaryMatrixZ{NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T}(Beeo, Beoo, Boeo, Booo) where {NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T} = SBoundaryMatrixZ{NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T}(nothing, Beeo, nothing, Beoo, nothing, Boeo, nothing, Booo)

abstract type BoundaryCondition end
struct BoundaryConditionZero <: BoundaryCondition
    M :: EvenOddProperty
end

struct BoundaryConditionSource <: BoundaryCondition
    M :: EvenOddProperty
    g :: BoundaryGridVariable
end

abstract type StaRMAPProblem end

struct DefaultStaRMAPProblem{T} <: StaRMAPProblem
    Ax::AdvectionMatrix
    Ay::AdvectionMatrix
    Az::AdvectionMatrix

    Bx::BoundaryCondition
    By::BoundaryCondition
    Bz::BoundaryCondition

    # S \partial_t u + sum A_i \partial_{x_i} u + C u = Q
    S::StaggeredGridVariable # capacity coefficient
    C::StaggeredGridVariable # transport coefficient (diagonal matrix)
    Q::StaggeredGridVariable # source term

    gspec::GridSpec
end

advection_matrix(problem::DefaultStaRMAPProblem, ::Type{X}) = problem.Ax
advection_matrix(problem::DefaultStaRMAPProblem, ::Type{Y}) = problem.Ay
advection_matrix(problem::DefaultStaRMAPProblem, ::Type{Z}) = problem.Az

boundary_condition(problem::DefaultStaRMAPProblem, ::Type{X}) = problem.Bx
boundary_condition(problem::DefaultStaRMAPProblem, ::Type{Y}) = problem.By
boundary_condition(problem::DefaultStaRMAPProblem, ::Type{Z}) = problem.Bz

capacity_coefficient(problem::DefaultStaRMAPProblem) = problem.S
homogeneous_term(problem::DefaultStaRMAPProblem) = problem.C
source_term(problem::DefaultStaRMAPProblem) = problem.Q

grid(problem::DefaultStaRMAPProblem) = problem.gspec


# function calc_du!(::Type{OddPairity}, problem::StaRMAPProblem, du, u, Δt, expm1div_)
#     dx = step(grid(problem)[X])
#     dy = step(grid(problem)[Y])
#     dz = step(grid(problem)[Z])

#     Ax = advection_matrix(problem, X)
#     Ay = advection_matrix(problem, Y)
#     Az = advection_matrix(problem, Z)

#     S = capacity_coefficient(problem)
#     C = homogeneous_term(problem)
#     Q = source_term(problem)

#     kernel!(du.eeo, Ax.eeo, Ay.eeo, Az.eeo, S.eeo, C.eeo, Q.eeo, u.oeo, u.eoo, u.eee, u.eeo, EvenOddClassification{Even, Even, Odd}, dx, dy, dz, Δt, expm1div_)
#     kernel!(du.eoe, Ax.eoe, Ay.eoe, Az.eoe, S.eoe, C.eoe, Q.eoe, u.ooe, u.eee, u.eoo, u.eoe, EvenOddClassification{Even, Odd, Even}, dx, dy, dz, Δt, expm1div_)
#     kernel!(du.oee, Ax.oee, Ay.oee, Az.oee, S.oee, C.oee, Q.oee, u.eee, u.ooe, u.oeo, u.oee, EvenOddClassification{Odd, Even, Even}, dx, dy, dz, Δt, expm1div_)
#     kernel!(du.ooo, Ax.ooo, Ay.ooo, Az.ooo, S.ooo, C.ooo, Q.ooo, u.eoo, u.oeo, u.ooe, u.ooo, EvenOddClassification{Odd, Odd, Odd}, dx, dy, dz, Δt, expm1div_)
# end
##

for dim in (X, Y, Z)
    for eop in (EvenPairity, OddPairity)
        eoc1, eoc2 = [eo for eo in all_eos_odd_in(dim) if pairity(eo) == eop]
        @eval begin
            function apply_boundary_!(::Type{$eop}, B::BoundaryConditionZero, u, ::Type{$dim})
                apply_boundary_zero!(u.$(to_symbol(eoc1)), u.$(to_symbol(switch_eo(eoc1, dim))), B.M.$(to_symbol(eoc1)), $dim)
                apply_boundary_zero!(u.$(to_symbol(eoc2)), u.$(to_symbol(switch_eo(eoc2, dim))), B.M.$(to_symbol(eoc2)), $dim)
            end
        end

        @eval begin
            function apply_boundary_!(::Type{$eop}, B::BoundaryConditionSource, u, ::Type{$dim})
                apply_boundary_source!(u.$(to_symbol(eoc1)), u.$(to_symbol(switch_eo(eoc1, dim))), B.M.$(to_symbol(eoc1)), B.g.$(to_symbol(eoc1)), $dim)
                apply_boundary_source!(u.$(to_symbol(eoc2)), u.$(to_symbol(switch_eo(eoc2, dim))), B.M.$(to_symbol(eoc2)), B.g.$(to_symbol(eoc2)), $dim)
            end
        end
    end
end

function apply_boundary!(eop::Type{<:EvenOddPairity}, problem::StaRMAPProblem, u)
    if !is_singleton(grid(problem)[X])
        Bx = boundary_condition(problem, X)
        apply_boundary_!(eop, Bx, u, X)
    end
    if !is_singleton(grid(problem)[Y])
        By = boundary_condition(problem, Y)
        apply_boundary_!(eop, By, u, Y)
    end
    if !is_singleton(grid(problem)[Z])
        Bz = boundary_condition(problem, Z)
        apply_boundary_!(eop, Bz, u, Z)
    end
end

function update_!(u, du, Δt)
    u .= u .+ Δt .* du
end

for eop in (EvenPairity, OddPairity)
    eoc1, eoc2, eoc3, eoc4 = [eo for eo in all_eos_of_eop(eop)]
    @eval begin
        function calc_du!(::Type{$eop}, problem::StaRMAPProblem, du, u, Δt, expm1div_)
            dx = step(grid(problem)[X])
            dy = step(grid(problem)[Y])
            dz = step(grid(problem)[Z])

            Ax = advection_matrix(problem, X)
            Ay = advection_matrix(problem, Y)
            Az = advection_matrix(problem, Z)

            S = capacity_coefficient(problem)
            C = homogeneous_term(problem)
            Q = source_term(problem)

            kernel!(du.$(to_symbol(eoc1)), Ax.$(to_symbol(eoc1)), Ay.$(to_symbol(eoc1)), Az.$(to_symbol(eoc1)), S.$(to_symbol(eoc1)), C.$(to_symbol(eoc1)), Q.$(to_symbol(eoc1)), u.$(to_symbol(switch_eo(eoc1, X))), u.$(to_symbol(switch_eo(eoc1, Y))), u.$(to_symbol(switch_eo(eoc1, Z))), u.$(to_symbol(eoc1)), $eoc1, dx, dy, dz, Δt, expm1div_)
            kernel!(du.$(to_symbol(eoc2)), Ax.$(to_symbol(eoc2)), Ay.$(to_symbol(eoc2)), Az.$(to_symbol(eoc2)), S.$(to_symbol(eoc2)), C.$(to_symbol(eoc2)), Q.$(to_symbol(eoc2)), u.$(to_symbol(switch_eo(eoc2, X))), u.$(to_symbol(switch_eo(eoc2, Y))), u.$(to_symbol(switch_eo(eoc2, Z))), u.$(to_symbol(eoc2)), $eoc2, dx, dy, dz, Δt, expm1div_)
            kernel!(du.$(to_symbol(eoc3)), Ax.$(to_symbol(eoc3)), Ay.$(to_symbol(eoc3)), Az.$(to_symbol(eoc3)), S.$(to_symbol(eoc3)), C.$(to_symbol(eoc3)), Q.$(to_symbol(eoc3)), u.$(to_symbol(switch_eo(eoc3, X))), u.$(to_symbol(switch_eo(eoc3, Y))), u.$(to_symbol(switch_eo(eoc3, Z))), u.$(to_symbol(eoc3)), $eoc3, dx, dy, dz, Δt, expm1div_)
            kernel!(du.$(to_symbol(eoc4)), Ax.$(to_symbol(eoc4)), Ay.$(to_symbol(eoc4)), Az.$(to_symbol(eoc4)), S.$(to_symbol(eoc4)), C.$(to_symbol(eoc4)), Q.$(to_symbol(eoc4)), u.$(to_symbol(switch_eo(eoc4, X))), u.$(to_symbol(switch_eo(eoc4, Y))), u.$(to_symbol(switch_eo(eoc4, Z))), u.$(to_symbol(eoc4)), $eoc4, dx, dy, dz, Δt, expm1div_)
         end
    end

    @eval begin
        function update!(::Type{$eop}, u, du, Δt)
            (n_x, n_y, n_z) = size(u.$(to_symbol(eoc1)))
            @inbounds for i in 1:n_x, j in 1:n_y, k in 1:n_z
                update_!(u.$(to_symbol(eoc1))[i, j, k], du.$(to_symbol(eoc1))[i, j, k], Δt)
            end
            (n_x, n_y, n_z) = size(u.$(to_symbol(eoc2)))
            @inbounds for i in 1:n_x, j in 1:n_y, k in 1:n_z
                update_!(u.$(to_symbol(eoc2))[i, j, k], du.$(to_symbol(eoc2))[i, j, k], Δt)
            end
            (n_x, n_y, n_z) = size(u.$(to_symbol(eoc3)))
            @inbounds for i in 1:n_x, j in 1:n_y, k in 1:n_z
                update_!(u.$(to_symbol(eoc3))[i, j, k], du.$(to_symbol(eoc3))[i, j, k], Δt)
            end
            (n_x, n_y, n_z) = size(u.$(to_symbol(eoc4)))
            @inbounds for i in 1:n_x, j in 1:n_y, k in 1:n_z
                update_!(u.$(to_symbol(eoc4))[i, j, k], du.$(to_symbol(eoc4))[i, j, k], Δt)
            end
        end
    end
end

function step_pde!(
    problem::StaRMAPProblem,
    u,
    du,
    t::Real,
    Δt::Real)

    # half step
    Δt_2 = Δt / 2
    apply_boundary!(EvenPairity, problem, u)
    calc_du!(OddPairity, problem, du, u, Δt_2, true)
    update!(OddPairity, u, du, Δt_2)

    # full step
    apply_boundary!(OddPairity, problem, u)
    calc_du!(EvenPairity, problem, du, u, Δt, true)
    update!(EvenPairity, u, du, Δt)

    # half step
    apply_boundary!(EvenPairity, problem, u)
    calc_du!(OddPairity, problem, du, u, Δt_2, true)
    update!(OddPairity, u, du, Δt_2)

    t = t + Δt
    return t
end
