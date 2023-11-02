struct NNPNParametrization{ODIM, IDIM, C} <: PNParametrization{ODIM, Float64}
    chain::C
end

NNPNParametrization{ODIM}(c::C) where {ODIM, C} = NNPNParametrization{ODIM, 3, C}(c)
NNPNParametrization{ODIM, IDIM}(c::C) where{ODIM, IDIM, C} = NNPNParametrization{ODIM, IDIM, C}(c)

Base.zero(m::NNPNParametrization{ODIM, IDIM}) where {ODIM, IDIM} = NNPNParametrization{ODIM, IDIM}(zero(m.chain))

n_out_dim(::NNPNParametrization{ODIM}) where ODIM = ODIM

n_params(m::NNPNParametrization) = n_params(m.chain)
Flux.trainable(m::NNPNParametrization) = Flux.trainable(m.chain)

to_named_tuple(p::NNPNParametrization) = to_named_tuple(p.chain)

function from_named_tuple!(p::NNPNParametrization, n::NamedTuple)
    from_named_tuple!(p.chain, n)
    return p
end

function param_vec!(p::AbstractVector, m::NNPNParametrization)
    @assert length(p) == n_params(m)
    param_vec!(p, m.chain)
end

function from_param_vec!(m::NNPNParametrization, p::AbstractVector)
    @assert length(p) == n_params(m)
    from_param_vec!(m.chain, p)
end

function parametrization!(ρ::AbstractVector, m::NNPNParametrization{ODIM, 3}, x::AbstractVector) where {ODIM}
    feed!(ρ, m.chain, x)
end

function parametrization!(ρ::AbstractVector, m::NNPNParametrization{ODIM, IDIM}, x::AbstractVector) where {ODIM, IDIM}
    feed!(ρ, m.chain, view(x, 1:IDIM))
end

function parametrization_pullback!(_, m::NNPNParametrization{ODIM, 3}, x::AbstractVector, ρ_::AbstractVector, m_::NNPNParametrization{ODIM, 3}) where {ODIM}
    feed!(nothing, m.chain, x)
    feed_pullback!(nothing, m.chain, x, ρ_, m_.chain)
end

function parametrization_pullback!(_, m::NNPNParametrization{ODIM, IDIM}, x::AbstractVector, ρ_::AbstractVector, m_::NNPNParametrization{ODIM, IDIM}) where {ODIM, IDIM}
    feed!(nothing, m.chain, view(x, 1:IDIM))
    feed_pullback!(nothing, m.chain, view(x, 1:IDIM), ρ_, m_.chain)
end

function parametrization_pullback_no_recompute!(_, m::NNPNParametrization{ODIM, 3}, x::AbstractVector, ρ_::AbstractVector, m_::NNPNParametrization{ODIM, 3}) where {ODIM}
    feed_pullback!(nothing, m.chain, x, ρ_, m_.chain)
end

function parametrization_pullback_no_recompute!(_, m::NNPNParametrization{ODIM, IDIM}, x::AbstractVector, ρ_::AbstractVector, m_::NNPNParametrization{ODIM, IDIM}) where {ODIM, IDIM}
    feed_pullback!(nothing, m.chain, view(x, 1:IDIM), ρ_, m_.chain)
end
