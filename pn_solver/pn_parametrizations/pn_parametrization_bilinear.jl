struct BilinearPNParametrization{OPTIMIZABLEEMBEDDING, ODIM, T, V, B, X, Y, Z} <: PNParametrization{ODIM, T}
    vals::V
    embedding::B
    x_slices::X
    y_slices::Y
    z_slices::Z
end

BilinearPNParametrization{true, ODIM}(vals::AbstractArray{<:AbstractVector{T}}, embedding::AbstractVector{T}, x_slices::X, y_slices::Y, z_slices::Z) where {ODIM, T, X<:StepRangeLen, Y<:StepRangeLen, Z<:StepRangeLen} = BilinearPNParametrization{true, ODIM, T, typeof(vals), typeof(embedding), X, Y, Z}(vals, embedding, x_slices, y_slices, z_slices)
BilinearPNParametrization{false, ODIM}(vals::AbstractArray{<:AbstractVector{T}}, embedding::B, x_slices::X, y_slices::Y, z_slices::Z) where {ODIM, T, B, X<:StepRangeLen, Y<:StepRangeLen, Z<:StepRangeLen} = BilinearPNParametrization{false, ODIM, T, typeof(vals), typeof(embedding), X, Y, Z}(vals, embedding, x_slices, y_slices, z_slices)

function Base.rand(::Type{BilinearPNParametrization{true, ODIM}}, x::Tuple, y::Tuple, z::Tuple, nx::Int, ny::Int, nz::Int) where {ODIM}
    return BilinearPNParametrization{true, ODIM}(
        [rand(ODIM) for i in 1:nx, j in 1:ny, k in 1:nz],
        rand(ODIM),
        range(x[1], x[2], length=nx),
        range(y[1], y[2], length=ny),
        range(z[1], z[2], length=nz))
end
function Base.rand(::Type{BilinearPNParametrization{false, ODIM}}, embedding::B, x::Tuple, y::Tuple, z::Tuple, nx::Int, ny::Int, nz::Int) where {ODIM, B}
    return BilinearPNParametrization{false, ODIM}(
        [rand(ODIM) for i in 1:nx, j in 1:ny, k in 1:nz],
        embedding,
        range(x[1], x[2], length=nx),
        range(y[1], y[2], length=ny),
        range(z[1], z[2], length=nz))
end
function Base.fill(::Type{BilinearPNParametrization{true, ODIM}}, x::Tuple, y::Tuple, z::Tuple, nx::Int, ny::Int, nz::Int, value) where {ODIM}
    return BilinearPNParametrization{true, ODIM}(
        [fill(value, ODIM) for i in 1:nx, j in 1:ny, k in 1:nz],
        fill(value, ODIM),
        range(x[1], x[2], length=nx),
        range(y[1], y[2], length=ny),
        range(z[1], z[2], length=nz))
end
function Base.fill(::Type{BilinearPNParametrization{false, ODIM}}, embedding::B, x::Tuple, y::Tuple, z::Tuple, nx::Int, ny::Int, nz::Int, value) where {ODIM, B}
    return BilinearPNParametrization{false, ODIM}(
        [fill(value, ODIM) for i in 1:nx, j in 1:ny, k in 1:nz],
        embedding,
        range(x[1], x[2], length=nx),
        range(y[1], y[2], length=ny),
        range(z[1], z[2], length=nz))
end

Base.zero(m::BilinearPNParametrization{true, ODIM}) where ODIM = BilinearPNParametrization{true, ODIM}([zero(m.vals[1]) for _ in m.vals], zero(m.embedding), deepcopy(m.x_slices), deepcopy(m.y_slices), deepcopy(m.z_slices))
Base.zero(m::BilinearPNParametrization{false, ODIM}) where ODIM  = BilinearPNParametrization{false, ODIM}([zero(m.vals[1]) for _ in m.vals], deepcopy(m.embedding), deepcopy(m.x_slices), deepcopy(m.y_slices), deepcopy(m.z_slices))
Base.rand(m::BilinearPNParametrization{true, ODIM}) where ODIM  = BilinearPNParametrization{true, ODIM}([rand(length(m.vals[1])) for _ in m.vals], rand(length(m.embedding)), deepcopy(m.x_slices), deepcopy(m.y_slices), deepcopy(m.z_slices))
Base.rand(m::BilinearPNParametrization{false, ODIM}) where ODIM = BilinearPNParametrization{false, ODIM}([rand(length(m.vals[1])) for _ in m.vals], deepcopy(m.embedding), deepcopy(m.x_slices), deepcopy(m.y_slices), deepcopy(m.z_slices))

n_params(m::BilinearPNParametrization{true}) = n_out_dims(m)*length(m.x_slices)*length(m.y_slices)*length(m.z_slices) + n_out_dims(m)
n_params(m::BilinearPNParametrization{false}) = n_out_dims(m)*length(m.x_slices)*length(m.y_slices)*length(m.z_slices)

n_out_dims(::BilinearPNParametrization{OE, ODIM}) where {OE, ODIM} = ODIM

Flux.trainable(m::BilinearPNParametrization{true}) = (m.vals..., m.embedding)
Flux.trainable(m::BilinearPNParametrization{false}) = (m.vals...,)

to_named_tuple(m::BilinearPNParametrization{true}) = (vals=m.vals, embedding=m.embedding)
to_named_tuple(m::BilinearPNParametrization{false}) = (vals=m.vals, )

function from_named_tuple!(m::BilinearPNParametrization{true}, p::NamedTuple)
    for (mv, pv) in zip(m.vals, p.vals)
        mv .= pv
    end
    m.embedding .= p.embedding
end
function from_named_tuple!(m::BilinearPNParametrization{false}, p::NamedTuple)
    for (mv, pv) in zip(m.vals, p.vals)
        mv .= pv
    end
end


function param_vec!(p::AbstractVector, m::BilinearPNParametrization{true})
    @assert length(p) == n_params(m)
    p .= vcat(m.vals..., m.embedding)
    return p
end

function param_vec!(p::AbstractVector, m::BilinearPNParametrization{false})
    @assert length(p) == n_params(m)
    p .= vcat(m.vals...,)
    return p
end

function from_param_vec!(m::BilinearPNParametrization{true}, p::AbstractVector)
    @assert length(p) == n_params(m)
    for a in 1:length(m.x_slices)*length(m.y_slices)*length(m.z_slices)
        m.vals[a] .= p[(a-1)*n_out_dims(m)+1 : (a)*n_out_dims(m)]
    end
    m.embedding .= p[end-n_out_dims(m)+1:end]
    return m
end

function from_param_vec(m::BilinearPNParametrization{true, ODIM}, p::AbstractVector{T}) where {ODIM, T}
    @assert length(p) == n_params(m)
    vals = [zeros(T, ODIM) for _ in m.vals]
    for a in 1:length(m.x_slices)*length(m.y_slices)*length(m.z_slices)
        vals[a] .= p[(a-1)*n_out_dims(m)+1 : (a)*n_out_dims(m)]
    end
    embedding = p[end-n_out_dims(m)+1:end]
    return BilinearPNParametrization{true, ODIM}(vals, embedding, m.x_slices, m.y_slices, m.z_slices)
end

function from_param_vec!(m::BilinearPNParametrization{false}, p::AbstractVector)
    @assert length(p) == n_params(m)
    for a in 1:length(m.x_slices)*length(m.y_slices)*length(m.z_slices)
        m.vals[a] .= p[(a-1)*n_out_dims(m)+1 : (a)*n_out_dims(m)]
    end
    return m
end

function from_param_vec(m::BilinearPNParametrization{false, ODIM}, p::AbstractVector{T}) where {ODIM, T}
    @assert length(p) == n_params(m)
    vals = [zeros(T, ODIM) for _ in m.vals]
    for a in 1:length(m.x_slices)*length(m.y_slices)*length(m.z_slices)
        vals[a] .= p[(a-1)*n_out_dims(m)+1 : (a)*n_out_dims(m)]
    end
    return BilinearPNParametrization{false, ODIM}(vals, deepcopy(m.embedding), m.x_slices, m.y_slices, m.z_slices)
end

function parametrization!(ρ::AbstractVector, m::BilinearPNParametrization, x::AbstractVector)
    bilinear_interpolation!(ρ, m.x_slices, m.y_slices, m.z_slices, m.vals, m.embedding, x[1], x[2], x[3])
end

function parametrization_pullback!(_, m::BilinearPNParametrization, x::AbstractVector, ρ_::AbstractVector, m_::BilinearPNParametrization)
    bilinear_interpolation!_pullback(m.x_slices, m.y_slices, m.z_slices, nothing, x[1], x[2], x[3], ρ_, m_.vals, m_.embedding)
end

parametrization_pullback_no_recompute!(_, m::BilinearPNParametrization, x::AbstractVector, ρ_::AbstractVector, m_::BilinearPNParametrization) =
    parametrization_pullback!(nothing, m, x, ρ_, m_)