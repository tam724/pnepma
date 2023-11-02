import Base.Iterators.ProductIterator
import Base.getindex
getindex(itr::ProductIterator, inds...) = ProductIterator(getindex.(itr.iterators, inds))
