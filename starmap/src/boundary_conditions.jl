# returns the "range" 1:1 (basically only the first index), because the dimension (second argument) and the boundary-dimension (third argument) are the same.
function left_range(n, ::Type{T}, ::Type{T}) where T<:Dimension
    return 1:1
end

# returns the "range" 1:n (means -> interate that dimension), because the dimension (second argument) and the boundary-dimension (third argument) are different.
function left_range(n, ::Type{<:Dimension}, ::Type{<:Dimension})
    return 1:n
end

function right_range(n, ::Type{T}, ::Type{T}) where T<:Dimension
    return n:n
end

function right_range(n, ::Type{<:Dimension}, ::Type{<:Dimension})
    return 1:n
end

function right_index(i, j, k, ::Type{X})
    return (i+1, j, k)
end

function right_index(i, j, k, ::Type{Y})
    return (i, j+1, k)
end

function right_index(i, j, k, ::Type{Z})
    return (i, j, k+1)
end

function left_index(i, j, k, ::Type{X})
    return (i-1, j, k)
end

function left_index(i, j, k, ::Type{Y})
    return (i, j-1, k)
end

function left_index(i, j, k, ::Type{Z})
    return (i, j, k-1)
end

function right_source(i, j, k, ::Type{X})
    return (2, j, k)
end

function right_source(i, j, k, ::Type{Y})
    return (i, 2, k)
end

function right_source(i, j, k, ::Type{Z})
    return (i, j, 2)
end

function left_source(i, j, k, ::Type{X})
    return (1, j, k)
end

function left_source(i, j, k, ::Type{Y})
    return (i, 1, k)
end

function left_source(i, j, k, ::Type{Z})
    return (i, j, 1)
end

function apply_boundary_source!(u::AbstractArray, v::AbstractArray, M::AbstractMatrix, g::AbstractArray, dim::Type{<:Dimension})
    # u is odd in dimension dim
    # v in even in dimension dim

    # u and v have same dimensions in != dim
    (n_x, n_y, n_z) = size(u)
    @inbounds for k in left_range(n_z, dim, Z), j in left_range(n_y, dim, Y), i in left_range(n_x, dim, X)
        mul!(u[i, j, k], M, v[i, j, k])
        u[i, j, k] .= 2. .* (.- u[i, j, k] .+ g[left_source(i, j, k, dim)...]) .- u[right_index(i, j, k, dim)...]
        #u[i, j, k] = 2. * (M * v[i, j, k] + g[left_source(i, j, k, dim)...]) - u[right_index(i, j, k, dim)...]
    end
    @inbounds for k in right_range(n_z, dim, Z), j in right_range(n_y, dim, Y), i in right_range(n_x, dim, X)
        mul!(u[i, j, k], M, v[left_index(i, j, k, dim)...])
        u[i, j, k] .= 2. .* (u[i, j, k] .+ g[right_source(i, j, k, dim)...]) .- u[left_index(i, j, k, dim)...]
        #u[i, j, k] = 2. * (-M * v[left_index(i, j, k, dim)...] + g[right_source(i, j, k, dim)...]) - u[left_index(i, j, k, dim)...]
    end
end

function apply_boundary_zero!(u::AbstractArray, v::AbstractArray, M::AbstractMatrix, dim::Type{<:Dimension})
    # u is odd in dimension dim
    # u is even in dimension dim

    # u and v have same dimension in != dim

    (n_x, n_y, n_z) = size(u)
    for k in left_range(n_z, dim, Z), j in left_range(n_y, dim, Y), i in left_range(n_x, dim, X)
        mul!(u[i, j, k], M, v[i, j, k])
        u[i, j, k] .= -2. .* u[i, j, k] .- u[right_index(i, j, k, dim)...]
        #u[i, j, k] = -2. * M * v[i, j, k] - u[right_index(i, j, k, dim)...]
    end
    for k in right_range(n_z, dim, Z), j in right_range(n_y, dim, Y), i in right_range(n_x, dim, X)
        mul!(u[i, j, k], M, v[left_index(i, j, k, dim)...])
        u[i, j, k] .= 2. .* u[i, j, k] .- u[left_index(i, j, k, dim)...] 
        #u[i, j, k] = 2. * M * v[left_index(i, j, k, dim)...] - u[left_index(i, j, k, dim)...]
    end
end

#
#
# function apply_boundary_x_source!(u::AbstractArray{T, 4}, v::AbstractArray{T, 4}, M::AbstractArray{T, 2}, g::AbstractArray{T, 4}) where T
#     (n_to, n_from) = size(M)
#     (n_mom_to, n_x_to, n_y_to, n_z_to) = size(u)
#     (n_mom_from, n_x_from, n_y_from, n_z_from) = size(v)
#     (n_mom_g, n_x_g, n_y_g, n_z_g) = size(g)
#
#     # u is odd in dimension x
#     # v is even in dimension x
#
#     @assert n_to == n_mom_to
#     @assert n_from == n_mom_from
#     @assert n_x_to == n_x_from + 1
#     @assert n_y_to == n_y_from
#     @assert n_z_to == n_z_from
#     @assert n_mom_g == n_mom_to
#     @assert n_x_g == 2
#     @assert n_y_g == n_y_to
#     @assert n_z_g == n_z_to
#
#     n_y = n_y_to
#     n_z = n_z_to
#
#     for j in 1:n_y
#         for k in 1:n_z
#             for m in 1:n_mom_to
#                 u[m, 1, j, k] = -u[m, 2, j, k] + 2*g[m, 1, j, k]
#                 u[m, n_x_to, j, k] = -u[m, n_x_to-1, j, k] + 2*g[m, 2, j, k]
#                 for n in 1:n_mom_from
#                     u[m, 1, j, k] += 2*(-M[m, n] * v[n, 1, j, k]) # left boundary
#                     u[m, n_x_to, j, k] += 2*(M[m, n] * v[n, n_x_from, j, k]) # right boundary
#                 end
#             end
#         end
#     end
# end
#
#
#
# function apply_boundary_x_zero!(u::AbstractArray{T, 4}, v::AbstractArray{T, 4}, M::AbstractArray{T, 2}) where T
#     (n_to, n_from) = size(M)
#     (n_mom_to, n_x_to, n_y_to, n_z_to) = size(u)
#     (n_mom_from, n_x_from, n_y_from, n_z_from) = size(v)
#
#     # u is odd in dimension x
#     # v is even in dimension x
#
#     @assert n_to == n_mom_to
#     @assert n_from == n_mom_from
#     @assert n_x_to == n_x_from + 1
#     @assert n_y_to == n_y_from
#     @assert n_z_to == n_z_from
#
#     n_y = n_y_to
#     n_z = n_z_to
#
#     for j in 1:n_y
#         for k in 1:n_z
#             for m in 1:n_mom_to
#                 u[m, 1, j, k] = -u[m, 2, j, k]
#                 u[m, n_x_to, j, k] = -u[m, n_x_to-1, j, k]
#                 for n in 1:n_mom_from
#                     u[m, 1, j, k] += 2*(-M[m, n] * v[n, 1, j, k]) # left boundary
#                     u[m, n_x_to, j, k] += 2*(M[m, n] * v[n, n_x_from, j, k]) # right boundary
#                 end
#             end
#         end
#     end
# end
#
# function apply_boundary_y_source!(u::AbstractArray{T, 4}, v::AbstractArray{T, 4}, M::AbstractArray{T, 2}, g::AbstractArray{T, 4}) where T
#     (n_to, n_from) = size(M)
#     (n_mom_to, n_x_to, n_y_to, n_z_to) = size(u)
#     (n_mom_from, n_x_from, n_y_from, n_z_from) = size(v)
#     (n_mom_g, n_x_g, n_y_g, n_z_g) = size(g)
#
#     # u is odd in dimension y
#     # v is even in dimension y
#
#     @assert n_to == n_mom_to
#     @assert n_from == n_mom_from
#     @assert n_x_to == n_x_from
#     @assert n_y_to == n_y_from + 1
#     @assert n_z_to == n_z_from
#     @assert n_mom_g == n_mom_to
#     @assert n_x_g == n_x_to
#     @assert n_y_g == 2
#     @assert n_z_g == n_z_to
#
#     n_x = n_x_to
#     n_z = n_z_to
#
#     for i in 1:n_x
#         for k in 1:n_z
#             for m in 1:n_mom_to
#                 u[m, i, 1, k] = -u[m, i, 2, k] + 2*g[m, i, 1, k]
#                 u[m, i, n_y_to, k] = -u[m, i, n_y_to - 1, k] + 2*g[m, i, 2, k]
#                 for n in 1:n_mom_from
#                     u[m, i, 1, k] += 2*(-M[m, n] * v[n, i, 1, k])
#                     u[m, i, n_y_to, k] += 2*(M[m, n] * v[n, i, n_y_from, k])
#                 end
#             end
#         end
#     end
# end
#
#
# function apply_boundary_y_zero!(u::AbstractArray{T, 4}, v::AbstractArray{T, 4}, M::AbstractArray{T, 2}) where T
#     (n_to, n_from) = size(M)
#     (n_mom_to, n_x_to, n_y_to, n_z_to) = size(u)
#     (n_mom_from, n_x_from, n_y_from, n_z_from) = size(v)
#
#     # u is odd in dimension y
#     # v is even in dimension y
#
#     @assert n_to == n_mom_to
#     @assert n_from == n_mom_from
#     @assert n_x_to == n_x_from
#     @assert n_y_to == n_y_from + 1
#     @assert n_z_to == n_z_from
#
#     n_x = n_x_to
#     n_z = n_z_to
#
#     for i in 1:n_x
#         for k in 1:n_z
#             for m in 1:n_mom_to
#                 u[m, i, 1, k] = -u[m, i, 2, k]
#                 u[m, i, n_y_to, k] = -u[m, i, n_y_to - 1, k]
#                 for n in 1:n_mom_from
#                     u[m, i, 1, k] += 2*(-M[m, n] * v[n, i, 1, k])
#                     u[m, i, n_y_to, k] += 2*(M[m, n] * v[n, i, n_y_from, k])
#                 end
#             end
#         end
#     end
# end
#
# function apply_boundary_z_source!(u::AbstractArray{T, 4}, v::AbstractArray{T, 4}, M::AbstractArray{T, 2}, g::AbstractArray{T, 4}) where T
#     (n_to, n_from) = size(M)
#     (n_mom_to, n_x_to, n_y_to, n_z_to) = size(u)
#     (n_mom_from, n_x_from, n_y_from, n_z_from) = size(v)
#     (n_mom_g, n_x_g, n_y_g, n_z_g) = size(g)
#
#     # u is odd in dimension z
#     # v is even in dimension z
#
#     @assert n_to == n_mom_to
#     @assert n_from == n_mom_from
#     @assert n_x_to == n_x_from
#     @assert n_y_to == n_y_from
#     @assert n_z_to == n_z_from + 1
#     @assert n_mom_g == n_mom_to
#     @assert n_x_g == n_x_to
#     @assert n_y_g == n_y_to
#     @assert n_z_g == 2
#
#     n_x = n_x_to
#     n_y = n_y_to
#
#     for i in 1:n_x
#         for j in 1:n_y
#             for m in 1:n_mom_to
#                 u[m, i, j, 1] = - u[m, i, j, 2] + 2*g[m, i, j, 1]
#                 u[m, i, j, n_z_to] = - u[m, i, j, n_z_to - 1] + 2*g[m, i, j, 2]
#                 for n in 1:n_mom_from
#                     u[m, i, j, 1] += 2*(-M[m, n]*v[n, i, j, 1])
#                     u[m, i, j, n_z_to] += 2*(M[m, n]*v[n, i, j, n_z_from])
#                 end
#             end
#         end
#     end
# end
#
# function apply_boundary_z_zero!(u::AbstractArray{T, 4}, v::AbstractArray{T, 4}, M::AbstractArray{T, 2}) where T
#     (n_to, n_from) = size(M)
#     (n_mom_to, n_x_to, n_y_to, n_z_to) = size(u)
#     (n_mom_from, n_x_from, n_y_from, n_z_from) = size(v)
#
#     # u is odd in dimension z
#     # v is even in dimension z
#
#     @assert n_to == n_mom_to
#     @assert n_from == n_mom_from
#     @assert n_x_to == n_x_from
#     @assert n_y_to == n_y_from
#     @assert n_z_to == n_z_from + 1
#
#     n_x = n_x_to
#     n_y = n_y_to
#
#     for i in 1:n_x
#         for j in 1:n_y
#             for m in 1:n_mom_to
#                 u[m, i, j, 1] = - u[m, i, j, 2]
#                 u[m, i, j, n_z_to] = - u[m, i, j, n_z_to - 1]
#                 for n in 1:n_mom_from
#                     u[m, i, j, 1] += 2*(-M[m, n]*v[n, i, j, 1])
#                     u[m, i, j, n_z_to] += 2*(M[m, n]*v[n, i, j, n_z_from])
#                 end
#             end
#         end
#     end
# end
