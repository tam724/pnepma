"""
    returns the iteration range for odd -> even arrays
    o  e  o  e  o  e  o
    1  1  2  2  3  3  4
       1     ->    n
"""
@inline function eo_range(::Type{Even}, n::Int)
    return 1:n
end

"""
    returns the iteration range for even -> odd arrays
    o  e  o  e  o  e  o
    1  1  2  2  3  3  4
          2  -> n-1
"""
@inline function eo_range(::Type{Odd}, n::Int)
    return 2:n-1
end

@inline function eo_index_left(::Type{Even}, i::Int)
    return i
end

@inline function eo_index_left(::Type{Odd}, i::Int)
    return i-1
end

@inline function eo_index_right(::Type{Even}, i::Int)
    return i+1
end

@inline function eo_index_right(::Type{Odd}, i::Int)
    return i
end

#TODO: for speed: consider precomputing exp1div prior to the 3-step-integration. it is reused in the two steps and takes quite some time (see solver.jl/step_pde!)
function expm1div(c)
    if abs(c) <= 0.0002
        return 1.0 + 0.5 * c + 1.0 / 6.0 * c^2 # + O(c^3)
    else
        return (exp(c) - 1) / c
    end
end

# function matmuldiff!(du::SVector, A::SparseMatrixCSC, u_right::SVector, u_left::SVector, dx)
#     du += A * (u_right - u_left)/dx
# end

# function matmuldiff!(du::SVector, A::SMatrix, u_right::SVector, u_left::SVector, dx)
#     du += A * (u_right - u_left)/dx
# end

# function matmuldiff!(du::SVector, A::Matrix, u_right::SVector, u_left::SVector, dx)
#     du += A * (u_right - u_left)/dx
# end

using SparseArrays:getcolptr
function matmuldiff!(du::AbstractVector, A::SparseMatrixCSC, u_right::AbstractVector, u_left::AbstractVector, dx)
    _, M = size(A)
    rows = rowvals(A)
    vals = nonzeros(A)
    @inbounds for m in 1:M
        d = (u_right[m] - u_left[m])/dx
        for n in getcolptr(A)[m]:(getcolptr(A)[m + 1] - 1)
            du[rows[n]] += vals[n]*d
        end
    end
end

# maybe non allocating implementation ?
function matmuldiff!(du::AbstractVector, A::AbstractMatrix, u_right::AbstractVector, u_left::AbstractVector, dx)
    du .+= A * (u_right .- u_left)./dx
end

# try to dispatch on the matrix A
function matmuldiff!(du::AbstractVector, ::Val{A}, u_right::AbstractVector, u_left::AbstractVector, dx) where A
    du .+= A * (u_right .- u_left)./dx
end

# non allocating version with full matrix A
function matmuldiff!(du::AbstractVector, A::Matrix, u_right::AbstractVector, u_left::AbstractVector, dx)
    N, M = size(A)
    @inbounds for m in 1:M
        du_ = (u_right[m] - u_left[m])/dx
        for n in 1:N
            du[n] += A[n, m] * du_
        end
    end
end

function right_hand_side!(du::AbstractVector, u_to::AbstractVector, Q::AbstractVector, C::AbstractVector, S::AbstractVector, Δt, expm1div_)
    if expm1div_    
        du .= (Q .- du .- C .* u_to) ./ S .* expm1div.(-C ./ S .* Δt)
    else
        du .= (Q .- du .- C .* u_to) ./ S
    end
end

function right_hand_side!(du::AbstractVector, u_to::AbstractVector, ::Zeros, C::AbstractVector, S::AbstractVector, Δt, expm1div_)
    if expm1div_
        du .= (.- du .- C .* u_to) ./ S .* expm1div.(-C ./ S .* Δt)
    else
        du .= (.- du .- C .* u_to) ./ S
    end
end

function kernel!(
    du::Array{<:AbstractArray{T}, 3},
    A_x,
    A_y,
    A_z,
    S::Array{<:AbstractArray, 3},
    C::Array{<:AbstractArray, 3},
    Q::Array{<:AbstractArray, 3},
    u_from_x::Array{<:AbstractArray, 3},
    u_from_y::Array{<:AbstractArray, 3},
    u_from_z::Array{<:AbstractArray, 3},
    u_to::Array{<:AbstractArray, 3},
    ::Type{EvenOddClassification{EOX, EOY, EOZ}},
    dx, dy, dz, Δt, expm1div_::Bool) where {T, EOX, EOY, EOZ}
    (n_x_to, n_y_to, n_z_to) = size(u_to)

    @inbounds for k in eo_range(EOZ, n_z_to), j in eo_range(EOY, n_y_to), i in eo_range(EOX, n_x_to)
        du[i, j, k] .= zero(du[i, j, k])
        if n_x_to != 1 # Singleton Dimension
            matmuldiff!(du[i, j, k], A_x, u_from_x[eo_index_right(EOX, i), j, k], u_from_x[eo_index_left(EOX, i), j, k], dx)
        end
        if n_y_to != 1 # Singleton Dimension
            matmuldiff!(du[i, j, k], A_y, u_from_y[i, eo_index_right(EOY, j), k], u_from_y[i, eo_index_left(EOY, j), k], dy)
        end
        if n_z_to != 1 # Singleton Dimension
            matmuldiff!(du[i, j, k], A_z, u_from_z[i, j, eo_index_right(EOZ, k)], u_from_z[i, j, eo_index_left(EOZ, k)], dz)
        end
        right_hand_side!(du[i, j, k], u_to[i, j, k], Q[i, j, k], C[i, j, k], S[i, j, k], Δt, expm1div_)
    end
end
