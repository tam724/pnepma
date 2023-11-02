abstract type Dimension end
struct X <: Dimension end
struct Y <: Dimension end
struct Z <: Dimension end

dim2idx(dim::Type{X}) = 1
dim2idx(dim::Type{Y}) = 2
dim2idx(dim::Type{Z}) = 3
