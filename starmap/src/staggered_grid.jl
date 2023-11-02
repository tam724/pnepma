import Base

## Material Dimensions with extent/ staggered properties
abstract type StaggeredDimension{T <: Real} end

struct EvenDimension{T <: Real, N <: Int} <: StaggeredDimension{T}
    number_of_gridpoints::N
    extent::Tuple{T, T}
end
#
# struct OddDimension{T <: Real} <: StaggeredDimension{T}
#     number_of_gridpoints::Int
#     extent::Tuple{T, T}
# end
#
# struct OddWithBoundaryDimension{T <: Real} <: StaggeredDimension{T}
#     number_of_gridpoints::Int
#     extent::Tuple{T, T}
# end

struct SingletonDimension{T <: Real} <: StaggeredDimension{T}
    position::T
end

struct OddWithGhostDimension{T  <: Real, N <: Int} <: StaggeredDimension{T}
    number_of_gridpoints::N
    extent::Tuple{T, T}
end

function Base.length(dim::SingletonDimension{T})::T where T
    return zero(T)
end

function Base.length(dim::StaggeredDimension{T})::T where T
    return dim.extent[2] - dim.extent[1]
end

function Base.size(dim::StaggeredDimension{T})::Int where T
    return dim.number_of_gridpoints
end

function Base.size(dim::SingletonDimension)::Int
    return 1
end

function Base.step(dim::EvenDimension{T})::T where T
    return length(dim) / (size(dim) - 1)
end

# function Base.step(dim::OddDimension{T})::T where T
#     return length(dim) / (size(dim) + 1)
# end

function Base.step(dim::OddWithGhostDimension{T})::T where T
    return length(dim) / (size(dim) - 2)
end

function Base.step(dim::SingletonDimension{T})::T where T
    return one(T)
end

# function Base.step(dim::OddWithBoundaryDimension{T})::Tuple{T, T} where T
#     h = length(dim) / (size(dim) - 2)
#     return (h, h/2)
# end

function points(dim::EvenDimension)
    return range(dim.extent[1], stop = dim.extent[2], length = size(dim))
end

function points(dim::SingletonDimension)
    return dim.position:dim.position
end

function points(dim::OddWithGhostDimension)
    h = step(dim)
    return range(dim.extent[1] - h / 2, stop=dim.extent[2] + h / 2, length=size(dim))
end

# function points(dim::OddDimension)
#     h = step(dim)
#     return range(
#         dim.extent[1] + h / 2,
#         stop = dim.extent[2] - h / 2,
#         length = dim.number_of_gridpoints,
#     )
# end
#
# function points(dim::OddWithBoundaryDimension)
#     (h, h_bound) = step(dim)
#     return [dim.extent[1]
#         range(dim.extent[1] + h / 2,
#         stop = dim.extent[2] - h / 2,
#         length = size(dim) - 2)
#         dim.extent[2]]
# end

# function left(dim::StaggeredDimension{T}) where T
#     return dim.extent[1]
# end

# function right(dim::StaggeredDimension{T}) where T
#     return dim.extent[2]
# end

## Dimension Pairs (e/o) for CFD finite difference operator
struct DimPair{T <: Real, E <: Union{EvenDimension{T}, SingletonDimension{T}}, O <: Union{OddWithGhostDimension{T}, SingletonDimension{T}}}
    e::E
    o::O
end

import Base.getindex
getindex(dim::DimPair, eo::Type{Even}) = dim.e
getindex(dim::DimPair, eo::Type{Odd}) = dim.o

function cfd_eo_switch(dim::EvenDimension)
    return OddWithGhostDimension(dim.number_of_gridpoints + 1, dim.extent)
end

function cfd_eo_switch(dim::OddWithGhostDimension)
    return EvenDimension(dim.number_of_gridpoints - 1, dim.extent)
end

function cfd_eo_switch(dim::SingletonDimension)
    return SingletonDimension(dim.position)
end

function make_dimension_pair(M::Int, ext::Tuple{<:Real, <:Real})
    e = EvenDimension(M, ext)
    o = cfd_eo_switch(e)
    @assert step(e) == step(o)
    return DimPair(e, o)
end

function make_dimension_pair(M::Int, pos::Real)
    e = SingletonDimension(pos)
    o = cfd_eo_switch(e)
    @assert step(e) == step(o)
    return DimPair(e, o)
end

function make_dimension_pairs(M::Int, x, y, z)
    return make_dimension_pair(M, x), make_dimension_pair(M, y), make_dimension_pair(M, z)
end

function make_dimension_pairs(M::AbstractArray{Int}, x, y, z)
    return make_dimension_pair(M[1], x), make_dimension_pair(M[2], y), make_dimension_pair(M[3], z)
end

function Base.step(dim::DimPair)
    return step(dim.e) # TODO: change !
end    

function is_singleton(dim::DimPair)
    return typeof(dim.e) <: SingletonDimension
end

## Grid Specification
struct GridSpec{T <: Real, D}
    dims::D
end

dx(gspec::GridSpec) = step(gspec[X])
dy(gspec::GridSpec) = step(gspec[Y])
dz(gspec::GridSpec) = step(gspec[Z])
dvol(gspec::GridSpec) = dx(gspec)*dy(gspec)*dz(gspec)

getindex(g::GridSpec, dim::Type{<:Dimension}) = g.dims[dim2idx(dim)]

function make_gridspec(M, x, y, z)
    dims = make_dimension_pair.(M, (x, y, z))
    return GridSpec{Float64, typeof(dims)}(dims)
    # return GridSpec(make_dimension_pair.(M, (x, y, z)))
end

## Staggered Grid (eee, eeo, etc.)
struct StaggeredGrid{T <: Real, D <: Tuple{<: StaggeredDimension{T}, <: StaggeredDimension{T}, <: StaggeredDimension{T}}}
    dimensions::D
end

StaggeredGrid(xdim::StaggeredDimension{T}, ydim::StaggeredDimension{T}, zdim::StaggeredDimension{T}) where T = StaggeredGrid((xdim, ydim, zdim))

function check_grid(a::StaggeredGrid, b::StaggeredGrid)
    if(any(a.dimensions .!= b.dimensions))
        n_mismatch_dim = findfirst(x->x, a.dimensions .!= b.dimensions)
        a_mismatch_dim = a.dimensions[n_mismatch_dim]
        b_mismatch_dim = b.dimensions[n_mismatch_dim]
        error("Dimension mismatch in dimension $n_mismatch_dim, $a_mismatch_dim != $b_mismatch_dim")
    end
end

# function Base.getindex(grid::StaggeredGrid, dim::Type{X})
#     return grid.dimensions[1]
# end
#
# function Base.getindex(grid::StaggeredGrid, dim::Type{Y})
#     return grid.dimensions[2]
# end
#
# function Base.getindex(grid::StaggeredGrid, dim::Type{Z})
#     return grid.dimensions[3]
# end

function points(eo::Type{EvenOddClassification{EOX, EOY, EOZ}}, gspec::GridSpec) where {EOX, EOY, EOZ}
    return points(grid(eo, gspec))
end

function points(grid::StaggeredGrid)
    return Iterators.product(points.(grid.dimensions)...)
end

# ##
# function points2(grid::StaggeredGrid)
#     return Iterators.product((points(grid.dimensions[1]), points(grid.dimensions[2]), points(grid.dimensions[3])))
# end
# ##

function Base.size(eo::Type{EvenOddClassification{EOX, EOY, EOZ}}, gspec::GridSpec) where {EOX, EOY, EOZ}
    return size(grid(eo, gspec))
end

function Base.size(eo::Type{EvenOddClassification{EOX, EOY, EOZ}}, gspec::GridSpec, i::Int) where {EOX, EOY, EOZ}
    return size(grid(eo, gspec), i)
end

function Base.size(grid::StaggeredGrid)
    return size.(grid.dimensions)
end

function Base.size(grid::StaggeredGrid, i::Int)
    return size(grid.dimensions[i])
end

## Grid Creation

function grid(::Type{EvenOddClassification{EOX, EOY, EOZ}}, xdim::DimPair, ydim::DimPair, zdim::DimPair) where {EOX, EOY, EOZ}
    return StaggeredGrid((xdim[EOX], ydim[EOY], zdim[EOZ]))
end

function grid(::Type{EvenOddClassification{EOX, EOY, EOZ}}, gspec::GridSpec) where {EOX, EOY, EOZ}
    return StaggeredGrid((gspec[X][EOX], gspec[Y][EOY], gspec[Z][EOZ]))
end
