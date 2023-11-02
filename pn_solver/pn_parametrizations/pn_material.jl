abstract type PNMaterial{NE, T} end
abstract type PNParametrization{ODIM, T} end

import Flux.trainable
import Base: rand, zero, adjoint

include("biliner_interp.jl")
include("nn_lib.jl")

include("pn_parametrization_const.jl")
include("pn_parametrization_bilinear.jl")
include("pn_parametrization_nn.jl")
include("pn_parametrization_boxed.jl")
include("pn_parametrization_test.jl")

##

n_elements(m::PNMaterial) = length(m.elms)

n_params(m::PNMaterial) = n_params(m.p)
Flux.trainable(m::PNMaterial) = Flux.trainable(m.p)

function from_named_tuple!(m::PNMaterial, p::NamedTuple)
    from_named_tuple!(m.p, p.p)
    return m
end

function param_vec!(p::AbstractVector, m::PNMaterial)
    param_vec!(p, m.p)
    return p
end

function param_vec(m::PNMaterial{NE, T}) where {NE, T}
    p = zeros(T, n_params(m))
    param_vec!(p, m)
    return p
end

function param_vec(p::PNParametrization{ODIM, T}) where {ODIM, T}
    pvec = zeros(T, n_params(p))
    param_vec!(pvec, p)
    return pvec
end

function from_param_vec!(m::PNMaterial, p::AbstractVector)
    from_param_vec!(m.p, p)
    return m
end

function from_param_vec(m::PNMaterial, p::AbstractVector)
    m_copy = zero(m)
    from_param_vec!(m_copy, p)
    return m_copy
end

## MassConcentrationMaterial

struct MassConcentrationPNMaterial{NE, T, E, P} <: PNMaterial{NE, T}
    elms::E
    p::P
end

MassConcentrationPNMaterial(elms::AbstractVector{Element}, p::PNParametrization{ODIM, T}) where {ODIM, T} = MassConcentrationPNMaterial{ODIM, T, typeof(elms), typeof(p)}(elms, p)
MassConcentrationPNMaterial(m::MassConcentrationPNMaterial, p::PNParametrization{ODIM, T}) where {ODIM, T} = MassConcentrationPNMaterial{ODIM, T, typeof(m.elms), typeof(p)}(m.elms, p)

Base.zero(m::MassConcentrationPNMaterial{NE, T, E, P}) where {NE, T, E, P} = MassConcentrationPNMaterial{NE, T, typeof(m.elms), typeof(m.p)}(m.elms, zero(m.p))
Base.rand(m::MassConcentrationPNMaterial{NE, T, E, P}) where {NE, T, E, P} = MassConcentrationPNMaterial{NE, T, typeof(m.elms), typeof(m.p)}(m.elms, rand(m.p))

to_named_tuple(m::MassConcentrationPNMaterial) = (elms=nothing, p=to_named_tuple(m.p))

function component_densities!(ρ::AbstractVector, m::MassConcentrationPNMaterial, x::AbstractVector)
    parametrization!(ρ, m.p, x)
end

function component_densities_pullback!(_, m::MassConcentrationPNMaterial, x::AbstractVector, ρ_::AbstractVector, m_::MassConcentrationPNMaterial)
    parametrization_pullback!(nothing, m.p, x, ρ_, m_.p)
end

## Material that parametrizes volume fraction \phi
struct VolumeFractionPNMaterial{NE, T, E, R, P} <: PNMaterial{NE, T}
    elms::E
    ρ_pure::R
    p::P
end

struct VolumeFractionPNMaterialSC{NE, T, E, R, P} <: PNMaterial{NE, T}
    elms::E
    ρ_pure::R
    p::P
end

function VolumeFractionPNMaterial(elms::AbstractVector{Element}, p::PNParametrization{ODIM, T}) where {ODIM, T}
    ρ = [convert_strip_density(e.density) for e in elms]
    return VolumeFractionPNMaterial{ODIM, T, typeof(elms), typeof(ρ), typeof(p)}(elms, ρ, p)
end

function VolumeFractionPNMaterialSC(elms::AbstractVector{Element}, p::PNParametrization{ODIM, T}) where {ODIM, T}
    ρ = [convert_strip_density(e.density) for e in elms]
    return VolumeFractionPNMaterialSC{ODIM+1, T, typeof(elms), typeof(ρ), typeof(p)}(elms, ρ, p)
end

VolumeFractionPNMaterial(m::VolumeFractionPNMaterial, p::PNParametrization) = VolumeFractionPNMaterial(m.elms, p)
VolumeFractionPNMaterialSC(m::VolumeFractionPNMaterialSC, p::PNParametrization) = VolumeFractionPNMaterialSC(m.elms, p)

Base.zero(m::VolumeFractionPNMaterial{NE, T, E, R, P}) where {NE, T, E, R, P} = VolumeFractionPNMaterial{NE, T, typeof(m.elms), typeof(m.ρ_pure), typeof(m.p)}(m.elms, m.ρ_pure, zero(m.p))
Base.rand(m::VolumeFractionPNMaterial{NE, T, E, R, P}) where {NE, T, E, R, P} = VolumeFractionPNMaterial{NE, T, typeof(m.elms), typeof(m.ρ_pure), typeof(m.p)}(m.elms, m.ρ_pure, rand(m.p))

Base.zero(m::VolumeFractionPNMaterialSC{NE, T, E, R, P}) where {NE, T, E, R, P} = VolumeFractionPNMaterialSC{NE, T, typeof(m.elms), typeof(m.ρ_pure), typeof(m.p)}(m.elms, m.ρ_pure, zero(m.p))
Base.rand(m::VolumeFractionPNMaterialSC{NE, T, E, R, P}) where {NE, T, E, R, P} = VolumeFractionPNMaterialSC{NE, T, typeof(m.elms), typeof(m.ρ_pure), typeof(m.p)}(m.elms, m.ρ_pure, rand(m.p))

to_named_tuple(m::VolumeFractionPNMaterial) = (elms=nothing, ρ_pure=nothing, p=to_named_tuple(m.p))
to_named_tuple(m::VolumeFractionPNMaterialSC) = (elms=nothing, ρ_pure=nothing, p=to_named_tuple(m.p))

function component_densities!(ρ::AbstractVector, m::VolumeFractionPNMaterial, x::AbstractVector)
    parametrization!(ρ, m.p, x)
    ρ .*= m.ρ_pure
end

function component_densities!(ρ::AbstractVector, m::VolumeFractionPNMaterialSC, x::AbstractVector)
    parametrization!(@view(ρ[1:end-1]), m.p, x)
    ρ[end] = 1. - sum(@view(ρ[1:end-1]))
    ρ .*= m.ρ_pure
end

function component_densities_pullback!(_, m::VolumeFractionPNMaterial, x::AbstractVector, ρ_::AbstractVector, m_::VolumeFractionPNMaterial)
    ρ_ .*= m.ρ_pure
    parametrization_pullback!(nothing, m.p, x, ρ_, m_.p)
end

function component_densities_pullback!(_, m::VolumeFractionPNMaterialSC, x::AbstractVector, ρ_::AbstractVector, m_::VolumeFractionPNMaterialSC)
    ρ_ .*= m.ρ_pure
    ρ_[1:end-1] .-= ρ_[end]
    parametrization_pullback!(nothing, m.p, x, @view(ρ_[1:end-1]), m_.p)
end
##

struct LinearDensityPNMaterial{NE, T, E, R, P} <: PNMaterial{NE, T}
    elms::E
    ρ::R
    p::P
end


struct LinearDensityPNMaterialSC{NE, T, E, R, P} <: PNMaterial{NE, T}
    elms::E
    ρ::R
    p::P
end

function LinearDensityPNMaterial(elms::AbstractVector{Element}, p::PNParametrization{ODIM, T}) where {ODIM, T}
    ρ = [convert_strip_density(e.density) for e in elms]
    LinearDensityPNMaterial{ODIM, T, typeof(elms), typeof(ρ), typeof(p)}(elms, ρ, p)
end

function LinearDensityPNMaterialSC(elms::AbstractVector{Element}, p::PNParametrization{ODIM, T}) where {ODIM, T}
    ρ = [convert_strip_density(e.density) for e in elms]
    LinearDensityPNMaterialSC{ODIM+1, T, typeof(elms), typeof(ρ), typeof(p)}(elms, ρ, p)
end

LinearDensityPNMaterial(m::LinearDensityPNMaterial, p::PNParametrization) = LinearDensityPNMaterial(m.elms, p)
LinearDensityPNMaterialSC(m::LinearDensityPNMaterialSC, p::PNParametrization) = LinearDensityPNMaterialSC(m.elms, p)

Base.zero(m::LinearDensityPNMaterial{NE, T, E, R, P}) where {NE, T, E, R, P} = LinearDensityPNMaterial{NE, T, typeof(m.elms), typeof(m.ρ), typeof(m.p)}(m.elms, m.ρ, zero(m.p))
Base.rand(m::LinearDensityPNMaterial{NE, T, E, R, P}) where {NE, T, E, R, P} = LinearDensityPNMaterial{NE, T, typeof(m.elms), typeof(m.ρ), typeof(m.p)}(m.elms, m.ρ, rand(m.p))

Base.zero(m::LinearDensityPNMaterialSC{NE, T, E, R, P}) where {NE, T, E, R, P} = LinearDensityPNMaterialSC{NE, T, typeof(m.elms), typeof(m.ρ), typeof(m.p)}(m.elms, m.ρ, zero(m.p))
Base.rand(m::LinearDensityPNMaterialSC{NE, T, E, R, P}) where {NE, T, E, R, P} = LinearDensityPNMaterialSC{NE, T, typeof(m.elms), typeof(m.ρ), typeof(m.p)}(m.elms, m.ρ, rand(m.p))

to_named_tuple(m::LinearDensityPNMaterial) = (elms=nothing, ρ=nothing, p=to_named_tuple(m.p))
to_named_tuple(m::LinearDensityPNMaterialSC) = (elms=nothing, ρ=nothing, p=to_named_tuple(m.p))

function component_densities!(ρ::AbstractVector, m::LinearDensityPNMaterial, x::AbstractVector)
    parametrization!(ρ, m.p, x)
    ρ .*= dot(ρ, m.ρ)
end

function component_densities!(ρ::AbstractVector, m::LinearDensityPNMaterialSC, x::AbstractVector)
    parametrization!(@view(ρ[1:end-1]), m.p, x)
    ρ[end] = 1. - sum(@view(ρ[1:end-1]))
    ρ .*= dot(ρ, m.ρ)
end

function component_densities_pullback!(_, m::LinearDensityPNMaterial, x::AbstractVector, ρ_::AbstractVector, m_::LinearDensityPNMaterial)
    p = zero(ρ_)
    parametrization!(p, m.p, x)
    p_ = ρ_ .* dot(p, m.ρ) .+ m.ρ .* dot(p, ρ_)
    parametrization_pullback_no_recompute!(nothing, m.p, x, p_, m_.p)
end

function component_densities_pullback!(_, m::LinearDensityPNMaterialSC, x::AbstractVector, ρ_::AbstractVector, m_::LinearDensityPNMaterialSC)
    p = zero(ρ_)
    parametrization!(@view(p[1:end-1]), m.p, x)
    p[end] = 1. - sum(@view(p[1:end-1]))
    p_ = ρ_ .* dot(p, m.ρ) .+ m.ρ .* dot(p, ρ_)
    p_[1:end-1] .-= p_[end]
    parametrization_pullback_no_recompute!(nothing, m.p, x, @view(p_[1:end-1]), m_.p)
end

## Mass Fraction Material
struct MassFractionPNMaterial{NE, T, E, R, P} <: PNMaterial{NE, T}
    elms::E
    ρ_pure::R
    p::P
end

struct MassFractionPNMaterialSC{NE, T, E, R, P} <: PNMaterial{NE, T}
    elms::E
    ρ_pure::R
    p::P
end

function MassFractionPNMaterial(elms::AbstractVector{Element}, p::PNParametrization{ODIM, T}) where {ODIM, T}
    ρ = [convert_strip_density(e.density) for e in elms]
    MassFractionPNMaterial{ODIM, T, typeof(elms), typeof(ρ), typeof(p)}(elms, ρ, p)
end
function MassFractionPNMaterialSC(elms::AbstractVector{Element}, p::PNParametrization{ODIM, T}) where {ODIM, T}
    ρ = [convert_strip_density(e.density) for e in elms]
    MassFractionPNMaterialSC{ODIM+1, T, typeof(elms), typeof(ρ), typeof(p)}(elms, ρ, p)
end

MassFractionPNMaterial(m::MassFractionPNMaterial, p::PNParametrization) = MassFractionPNMaterial(m.elms, p)
MassFractionPNMaterialSC(m::MassFractionPNMaterialSC, p::PNParametrization) = MassFractionPNMaterialSC(m.elms, p)

Base.zero(m::MassFractionPNMaterial{NE, T, E, R, P}) where {NE, T, E, R, P} = MassFractionPNMaterial{NE, T, typeof(m.elms), typeof(m.ρ_pure), typeof(m.p)}(m.elms, m.ρ_pure, zero(m.p))
Base.rand(m::MassFractionPNMaterial{NE, T, E, R, P}) where {NE, T, E, R, P} = MassFractionPNMaterial{NE, T, typeof(m.elms), typeof(m.ρ_pure), typeof(m.p)}(m.elms, m.ρ_pure, rand(m.p))

Base.zero(m::MassFractionPNMaterialSC{NE, T, E, R, P}) where {NE, T, E, R, P} = MassFractionPNMaterialSC{NE, T, typeof(m.elms), typeof(m.ρ_pure), typeof(m.p)}(m.elms, m.ρ_pure, zero(m.p))
Base.rand(m::MassFractionPNMaterialSC{NE, T, E, R, P}) where {NE, T, E, R, P} = MassFractionPNMaterialSC{NE, T, typeof(m.elms), typeof(m.ρ_pure), typeof(m.p)}(m.elms, m.ρ_pure, rand(m.p))

to_named_tuple(m::MassFractionPNMaterial) = (elms=nothing, ρ_pure=nothing, p=to_named_tuple(m.p))
to_named_tuple(m::MassFractionPNMaterialSC) = (elms=nothing, ρ_pure=nothing, p=to_named_tuple(m.p))

function component_densities!(ρ::AbstractVector, m::MassFractionPNMaterial, x::AbstractVector)
    parametrization!(ρ, m.p, x)
    R = sum(ρ ./ m.ρ_pure)
    ρ ./= R
end

function component_densities!(ρ::AbstractVector, m::MassFractionPNMaterialSC, x::AbstractVector)
    parametrization!(@view(ρ[1:end-1]), m.p, x)
    ρ[end] = 1. - sum(@view(ρ[1:end-1]))
    R = sum(ρ ./ m.ρ_pure)
    ρ ./= R
end

function component_densities_pullback!(_, m::MassFractionPNMaterial, x::AbstractVector, ρ_::AbstractVector, m_::MassFractionPNMaterial)
    p = zero(ρ_)
    parametrization!(p, m.p, x)
    R = sum(p ./ m.ρ_pure)
    R_ = -dot(p, ρ_)/R^2
    p_ = ρ_ ./ R + R_ ./ m.ρ_pure
    parametrization_pullback_no_recompute!(nothing, m.p, x, p_, m_.p)
end

function component_densities_pullback!(_, m::MassFractionPNMaterialSC, x::AbstractVector, ρ_::AbstractVector, m_::MassFractionPNMaterialSC)
    p = zero(ρ_)
    parametrization!(@view(p[1:end-1]), m.p, x)
    p[end] = 1. - sum(@view(p[1:end-1]))
    R = sum(p ./ m.ρ_pure)
    R_ = -dot(p, ρ_)/R^2
    p_ = ρ_ ./ R + R_ ./ m.ρ_pure
    p_[1:end-1] .-= p_[end]
    parametrization_pullback_no_recompute!(nothing, m.p, x, @view(p_[1:end-1]), m_.p)
end

include("pn_material_utils.jl")