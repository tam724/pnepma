struct ConstPNParametrization{ODIM, P, T} <: PNParametrization{ODIM, T}
    p::P
end

ConstPNParametrization{ODIM}(p::AbstractVector{T}) where {ODIM, T} = ConstPNParametrization{ODIM, typeof(p), T}(p)
# ConstPNParametrization{ODIM, PO, TO}(::ConstPNParametrization{ODIM}, p::AbstractVector{T}) where {ODIM, T, PO, TO} = ConstPNParametrization{ODIM, typeof(p), T}(p)

Base.zero(m::ConstPNParametrization{ODIM}) where ODIM = ConstPNParametrization{ODIM}(zero(m.p))
Base.rand(m::ConstPNParametrization{ODIM}) where ODIM = ConstPNParametrization{ODIM}(rand(length(m.p)))

n_out_dims(::ConstPNParametrization{ODIM}) where ODIM = ODIM

n_params(m::ConstPNParametrization) = n_out_dims(m)
Flux.trainable(m::ConstPNParametrization) = (m.p, )

to_named_tuple(m::ConstPNParametrization) = (p=m.p, )
from_named_tuple!(m::ConstPNParametrization, p::NamedTuple) = m.p .= p.p

function param_vec!(p::AbstractVector, m::ConstPNParametrization)
    p .= m.p
    return p
end

function from_param_vec!(m::ConstPNParametrization, p::AbstractVector)
    m.p .= p
    return m
end

function from_param_vec(::ConstPNParametrization{ODIM}, p::AbstractVector{T}) where {ODIM, T}
    return ConstPNParametrization{ODIM}(p)
end

function parametrization!(ρ::AbstractVector, m::ConstPNParametrization, _::AbstractVector)
    ρ .= m.p
end

function parametrization_pullback!(_, ::ConstPNParametrization, ::AbstractVector,  ρ_::AbstractVector, p_::ConstPNParametrization)
    p_.p .+= ρ_
end

parametrization_pullback_no_recompute!(_, p::ConstPNParametrization, x::AbstractVector,  ρ_::AbstractVector, p_::ConstPNParametrization) =
    parametrization_pullback!(nothing, p, x, ρ_, p_)
