struct Moment
    degree::Int64
    order::Int64

    function Moment(degree::Int64, order::Int64)
        if degree < 0 @assert(false, "degree is negative") end
        if abs(order) > degree @assert(false, "abs(order) <= degree") end
        return new(degree, order)
    end
end

degree(m::Moment) = m.degree
order(m::Moment) = m.order

import Base: iterate
function iterate(m::Moment, i=1)
    if i == 1
        return (degree(m), i+1)
    elseif i == 2
        return (order(m), i+1)
    else
        return nothing
    end
end

function is_even_odd(m::Moment, ::Type{X})::Type{<:EvenOdd}
    return (m.order < 0 && isodd(m.order)) || (m.order >= 0 && iseven(m.order)) ? Even : Odd
end

function is_even_odd(m::Moment, ::Type{Y})::Type{<:EvenOdd}
    return m.order >= 0 ? Even : Odd
end

function is_even_odd(m::Moment, ::Type{Z})::Type{<:EvenOdd}
    return iseven(m.degree + m.order) ? Even : Odd
end

function full_eo_classification(m::Moment)
    return EvenOddClassification{map(dim -> is_even_odd(m, dim), (X, Y, Z))...}
end

function all_moments(Nmax::Int)
    return [Moment(l, k) for l in 0:Nmax for k in -l:l]
end

function ordered_moments(Nmax::Int)
    all = all_moments(Nmax)
    return [
        [m for m in all if full_eo_classification(m) == EvenOddClassification{Even, Even, Even}]...,
        [m for m in all if full_eo_classification(m) == EvenOddClassification{Even, Odd, Odd}]...,
        [m for m in all if full_eo_classification(m) == EvenOddClassification{Odd, Even, Odd}]...,
        [m for m in all if full_eo_classification(m) == EvenOddClassification{Odd, Odd, Even}]...,
        [m for m in all if full_eo_classification(m) == EvenOddClassification{Even, Even, Odd}]...,
        [m for m in all if full_eo_classification(m) == EvenOddClassification{Even, Odd, Even}]...,
        [m for m in all if full_eo_classification(m) == EvenOddClassification{Odd, Even, Even}]...,
        [m for m in all if full_eo_classification(m) == EvenOddClassification{Odd, Odd, Odd}]...
    ]
end