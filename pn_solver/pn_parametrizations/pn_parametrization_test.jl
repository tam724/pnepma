## used to generate measurements - for comparison with other discretizations

using Distances

struct FunctionPNParametrization{ODIM, F} <: PNParametrization{ODIM, Float64}
    f::F
end

FunctionPNParametrization{ODIM}(f::F) where {ODIM, F} = FunctionPNParametrization{ODIM, F}(f)

function parametrization!(ρ::AbstractVector, m::FunctionPNParametrization, x::AbstractVector)
    ρ .= m.f(x)
end

function parametrization_pullback!(ρ, m::FunctionPNParametrization, x::AbstractVector, ρ_::AbstractVector, m_::FunctionPNParametrization)
    error("not implemented")
end

struct InclusionPNParametrization{ODIM, X, Y, R, V, O} <: PNParametrization{ODIM, Float64}
    x::X
    y::Y
    r::R
    vals_inside::V
    vals_outside::O
end

InclusionPNParametrization{ODIM}(x::X, r::R, vals_inside::V, vals_outside::O) where {ODIM, X, R, V, O} = InclusionPNParametrization{ODIM, X, Diagonal{Bool, Vector{Bool}}, R, V, O}(x, I(3), r, vals_inside, vals_outside)
InclusionPNParametrization{ODIM}(x::X, y::Y, r::R, vals_inside::V, vals_outside::O) where {ODIM, Y, X, R, V, O} = InclusionPNParametrization{ODIM, X, Y, R, V, O}(x, y, r, vals_inside, vals_outside)

function parametrization!(p::AbstractVector, m::InclusionPNParametrization, x::AbstractVector)
    if sqrt((x - m.x)' * m.y * (x - m.x))  < m.r
        p .= m.vals_inside
    else
        p .= m.vals_outside
    end
end