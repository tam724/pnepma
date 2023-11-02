struct BoxedPNParametrization{OPTIMIZABLEEMBEDDING, ODIM, T, V, B, X, Y, Z} <: PNParametrization{ODIM, T}
    vals::V
    embedding::B
    x_slices::X
    y_slices::Y
    z_slices::Z
end

BoxedPNParametrization{true, ODIM}(vals::AbstractArray{<:AbstractVector{T}}, embedding::AbstractVector{T}, x_slices::X, y_slices::Y, z_slices::Z) where {ODIM, T, X, Y, Z} = BoxedPNParametrization{true, ODIM, T, typeof(vals), typeof(embedding), X, Y, Z}(vals, embedding, x_slices, y_slices, z_slices)
BoxedPNParametrization{false, ODIM}(vals::AbstractArray{<:AbstractVector{T}}, embedding::B, x_slices::X, y_slices::Y, z_slices::Z) where {ODIM, T, B, X, Y, Z} = BoxedPNParametrization{false, ODIM, T, typeof(vals), typeof(embedding), X, Y, Z}(vals, embedding, x_slices, y_slices, z_slices)

function Base.rand(::Type{BoxedPNParametrization{true, ODIM}}, x::Tuple, y::Tuple, z::Tuple, nx::Int, ny::Int, nz::Int) where {ODIM}
    return BoxedPNParametrization{true, ODIM}(
        [rand(ODIM) for i in 1:nx, j in 1:ny, k in 1:nz],
        rand(ODIM),
        range(x[1], x[2], length=nx+1),
        range(y[1], y[2], length=ny+1),
        range(z[1], z[2], length=nz+1)
    )
end
function Base.rand(::Type{BoxedPNParametrization{false, ODIM}}, embedding::B, x::Tuple, y::Tuple, z::Tuple, nx::Int, ny::Int, nz::Int) where {ODIM, B}
    return BoxedPNParametrization{false, ODIM}(
        [rand(ODIM) for i in 1:nx, j in 1:ny, k in 1:nz],
        embedding,
        range(x[1], x[2], length=nx+1),
        range(y[1], y[2], length=ny+1),
        range(z[1], z[2], length=nz+1)
    )
end
function Base.fill(::Type{BoxedPNParametrization{true, ODIM}}, x::Tuple, y::Tuple, z::Tuple, nx::Int, ny::Int, nz::Int, value) where {ODIM}
    return BoxedPNParametrization{true, ODIM}(
        [fill(value, ODIM) for i in 1:nx, j in 1:ny, k in 1:nz],
        fill(value, ODIM),
        range(x[1], x[2], length=nx+1),
        range(y[1], y[2], length=ny+1),
        range(z[1], z[2], length=nz+1)
    )
end
function Base.fill(::Type{BoxedPNParametrization{false, ODIM}}, embedding::B, x::Tuple, y::Tuple, z::Tuple, nx::Int, ny::Int, nz::Int, value) where {ODIM, B}
    return BoxedPNParametrization{false, ODIM}(
        [fill(value, ODIM) for i in 1:nx, j in 1:ny, k in 1:nz],
        embedding,
        range(x[1], x[2], length=nx+1),
        range(y[1], y[2], length=ny+1),
        range(z[1], z[2], length=nz+1)
    )
end
function Base.fill(::Type{BoxedPNParametrization{true, ODIM}}, x_slices::AbstractVector, y_slices::AbstractVector, z_slices::AbstractVector, value) where {ODIM}
    return BoxedPNParametrization{true, ODIM}(
        [fill(value, ODIM) for i in 1:length(x_slices)-1, j in 1:length(y_slices)-1, k in 1:length(z_slices)-1],
        fill(value, ODIM),
        x_slices, y_slices, z_slices
    )
end
function Base.fill(::Type{BoxedPNParametrization{false, ODIM}}, embedding::B, x_slices::AbstractVector, y_slices::AbstractVector, z_slices::AbstractVector, value) where {ODIM, B}
    return BoxedPNParametrization{false, ODIM}(
        [fill(value, ODIM) for i in 1:length(x_slices)-1, j in 1:length(y_slices)-1, k in 1:length(z_slices)-1],
        embedding,
        x_slices, y_slices, z_slices
    )
end


Base.zero(m::BoxedPNParametrization{true, ODIM}) where ODIM = BoxedPNParametrization{true, ODIM}([zero(m.vals[1]) for _ in m.vals], zero(m.embedding), deepcopy(m.x_slices), deepcopy(m.y_slices), deepcopy(m.z_slices))
Base.zero(m::BoxedPNParametrization{false, ODIM}) where ODIM = BoxedPNParametrization{false, ODIM}([zero(m.vals[1]) for _ in m.vals], deepcopy(m.embedding), deepcopy(m.x_slices), deepcopy(m.y_slices), deepcopy(m.z_slices))
Base.rand(m::BoxedPNParametrization{true, ODIM}) where ODIM = BoxedPNParametrization{true, ODIM}([rand(length(m.vals[1])) for _ in m.vals], rand(length(m.embedding)), deepcopy(m.x_slices), deepcopy(m.y_slices), deepcopy(m.z_slices))
Base.rand(m::BoxedPNParametrization{false, ODIM}) where ODIM = BoxedPNParametrization{false, ODIM}([rand(length(m.vals[1])) for _ in m.vals], deepcopy(m.embedding), deepcopy(m.x_slices), deepcopy(m.y_slices), deepcopy(m.z_slices))

n_params(m::BoxedPNParametrization{true}) = n_out_dims(m)*(length(m.x_slices)-1)*(length(m.y_slices)-1)*(length(m.z_slices)-1) + n_out_dims(m)
n_params(m::BoxedPNParametrization{false}) = n_out_dims(m)*(length(m.x_slices)-1)*(length(m.y_slices)-1)*(length(m.z_slices)-1)

n_out_dims(::BoxedPNParametrization{OE, ODIM}) where {OE, ODIM} = ODIM

Flux.trainable(m::BoxedPNParametrization{true}) = (m.vals..., m.embedding)
Flux.trainable(m::BoxedPNParametrization{false}) = (m.vals...,)

to_named_tuple(m::BoxedPNParametrization{true}) = (vals=m.vals, embedding=m.embedding)
to_named_tuple(m::BoxedPNParametrization{false}) = (vals=m.vals, )

function from_named_tuple!(m::BoxedPNParametrization{true}, p::NamedTuple)
    for (mv, pv) in zip(m.vals, p.vals)
        mv .= pv
    end
    m.embedding .= p.embedding
end
function from_named_tuple!(m::BoxedPNParametrization{false}, p::NamedTuple)
    for (mv, pv) in zip(m.vals, p.vals)
        mv .= pv
    end
end

function param_vec!(p::AbstractVector, m::BoxedPNParametrization{true})
    @assert length(p) == n_params(m)
    p .= vcat(m.vals..., m.embedding)
    return p
end
function param_vec!(p::AbstractVector, m::BoxedPNParametrization{false})
    @assert length(p) == n_params(m)
    p .= vcat(m.vals...,)
    return p
end

function from_param_vec!(m::BoxedPNParametrization{true}, p::AbstractVector)
    @assert length(p) == n_params(m)
    for i in 1:length(m.vals)
        m.vals[i] .= p[(i-1)*n_out_dims(m) + 1:i*n_out_dims(m)]
    end
    m.embedding .= p[end-n_out_dims(m)+1:end]
    return m
end
function from_param_vec(m::BoxedPNParametrization{true, ODIM}, p::AbstractVector{T}) where {ODIM, T}
    @assert length(p) == n_params(m)
    vals = [zeros(T, ODIM) for _ in m.vals]
    for i in 1:length(m.vals)
        vals[i] .= p[(i-1)*n_out_dims(m) + 1:i*n_out_dims(m)]
    end
    embedding = p[end-n_out_dims(m)+1:end]
    return BoxedPNParametrization{true, ODIM}(vals, embedding, m.x_slices, m.y_slices, m.z_slices)
end

function from_param_vec!(m::BoxedPNParametrization{false}, p::AbstractVector)
    @assert length(p) == n_params(m)
    for i in 1:length(m.vals)
        m.vals[i] .= p[(i-1)*n_out_dims(m) + 1:i*n_out_dims(m)]
    end
    return m
end
function from_param_vec(m::BoxedPNParametrization{false, ODIM}, p::AbstractVector{T}) where {ODIM, T}
    @assert length(p) == n_params(m)
    vals = [zeros(T, ODIM) for _ in m.vals]
    for i in 1:length(m.vals)
        vals[i] .= p[(i-1)*n_out_dims(m) + 1:i*n_out_dims(m)]
    end
    return BoxedPNParametrization{false, ODIM}(vals, deepcopy(m.embedding), m.x_slices, m.y_slices, m.z_slices)
end

function parametrization!(ρ::AbstractVector, m::BoxedPNParametrization, x)
    i, dx = find_index_smaller_and_partition(m.x_slices, x[1])
    j, dy = find_index_smaller_and_partition(m.y_slices, x[2])
    k, dz = find_index_smaller_and_partition(m.z_slices, x[3])
    if i == length(m.x_slices) && isapproxzero(dx)
        i = i - 1
    end
    if j == length(m.y_slices) && isapproxzero(dy)
        j = j - 1
    end
    if k == length(m.z_slices) && isapproxzero(dz)
        k = k - 1
    end
    if i < 1 || i > length(m.x_slices) - 1 || j < 1 || j > length(m.y_slices) - 1 || k < 1 || k > length(m.z_slices) - 1
        ρ .= m.embedding
    else
        ρ .= m.vals[i, j, k]
    end
end

function parametrization_pullback!(_, m::BoxedPNParametrization, x::AbstractVector, ρ_::AbstractVector, m_::BoxedPNParametrization)
    i, dx = find_index_smaller_and_partition(m.x_slices, x[1])
    j, dy = find_index_smaller_and_partition(m.y_slices, x[2])
    k, dz = find_index_smaller_and_partition(m.z_slices, x[3])
    if i == length(m.x_slices) && isapproxzero(dx)
        i = i - 1
    end
    if j == length(m.y_slices) && isapproxzero(dy)
        j = j - 1
    end
    if k == length(m.z_slices) && isapproxzero(dz)
        k = k - 1
    end
    if i < 1 || i > length(m.x_slices) - 1 || j < 1 || j > length(m.y_slices) - 1 || k < 1 || k > length(m.z_slices) - 1
        m_.embedding .+= ρ_ 
    else
        m_.vals[i, j, k] .+= ρ_
    end
end

parametrization_pullback_no_recompute!(_, m::BoxedPNParametrization, x::AbstractVector, ρ_::AbstractVector, m_::BoxedPNParametrization) =
    parametrization_pullback!(nothing, m, x, ρ_, m_)