## advection matrices
function Θ(k::Int)::Int
    if k < 0
        return -1
    elseif k >= 0
        return 1
    end
end

function plus(k::Int)::Int
    if k == 0
        return 1
    end
    return k + sign(k)
end

function minus(k::Int)::Int
    if k == 0
        return -1
    end
    return k - sign(k)
end

function a(k::Int, l::Int)::Float64
    return sqrt((l - k + 1) * (l + k + 1) / ((2 * l + 3) * (2 * l + 1)))
end

function b(k::Int, l::Int)::Float64
    return sqrt((l - k) * (l + k) / ((2 * l + 1) * (2 * l - 1)))
end

function c(k::Int, l::Int)::Float64
    c = sqrt((l + k + 1) * (l + k + 2) / ((2 * l + 3) * (2 * l + 1)))
    if k < 0
        return 0
    elseif k > 0
        return c
    else
        return c * sqrt(2)
    end
end

function d(k::Int, l::Int)::Float64
    d = sqrt((l - k) * (l - k - 1) / ((2 * l + 1) * (2 * l - 1)))
    if k < 0
        return 0
    elseif k > 0
        return d
    else
        return d * sqrt(2)
    end
end

function e(k::Int, l::Int)::Float64
    e = sqrt((l - k + 1) * (l - k + 2) / ((2 * l + 3) * (2 * l + 1)))
    if k == 1
        return e * sqrt(2)
    elseif k > 1
        return e
    else
        error("k < 1")
    end
end

function f(k::Int, l::Int)::Float64
    f = sqrt((l + k) * (l + k - 1) / ((2 * l + 1) * (2 * l - 1)))
    if k == 1
        return f * sqrt(2)
    elseif k > 1
        return f
    else
        error("k < 1")
    end
end

function A_minus(l::Int, k::Int, k´::Int, dim::Type{X})::Float64
    if k´ == minus(k) && k != -1
        return 1 / 2 * c(abs(k) - 1, l - 1)
    elseif k´ == plus(k)
        return -1 / 2 * e(abs(k) + 1, l - 1)
    else
        return 0
    end
end

function A_minus(l::Int, k::Int, k´::Int, dim::Type{Y})::Float64
    if k´ == -minus(k) && k != 1
        return -Θ(k) / 2 * c(abs(k) - 1, l - 1)
    elseif k´ == -plus(k)
        return -Θ(k) / 2 * e(abs(k) + 1, l - 1)
    else
        return 0
    end
end

function A_minus(l::Int, k::Int, k´::Int, dim::Type{Z})::Float64
    if k´ == k
        return a(k, l - 1)
    else
        return 0
    end
end

function A_plus(l::Int, k::Int, k´::Int, dim::Type{X})::Float64
    if k´ == minus(k) && k != -1
        return -1 / 2 * d(abs(k) - 1, l + 1)
    elseif k´ == plus(k)
        return 1 / 2 * f(abs(k) + 1, l + 1)
    else
        return 0
    end
end

function A_plus(l::Int, k::Int, k´::Int, dim::Type{Y})::Float64
    if k´ == -minus(k) && k != 1
        return Θ(k) / 2 * d(abs(k) - 1, l + 1)
    elseif k´ == -plus(k)
        return Θ(k) / 2 * f(abs(k) + 1, l + 1)
    else
        return 0
    end
end

function A_plus(l::Int, k::Int, k´::Int, dim::Type{Z})::Float64
    if k´ == k
        return b(k, l + 1)
    else
        return 0
    end
end

function get_coefficient(m_to, m_from, dim)::Float64
    l, k = m_to
    l´, k´ = m_from
    if l == l´- 1
        # take Aplus
        return A_plus(l, k, k´, dim)
    elseif l == l´ + 1
        # take Aminus
        return A_minus(l, k, k´, dim)
    else
        return 0
    end
end

function make_advection_matrix(::Type{T}, moments_to::Array{Moment, 1}, moments_from::Array{Moment, 1}, dim::Type{<:Dimension}) where T
    A = spzeros(T, length(moments_to), length(moments_from))
    for (i, m_to) in enumerate(moments_to), (j, m_from) in enumerate(moments_from)
        A[i, j] = get_coefficient(m_to, m_from, dim)
    end
    return A
end

##

# function advection_matrix(::PNProblem{T, N}, ::Type{X}) where {T, N}
#     moments = make_moments(N)
#     return SAdvectionMatrixX{N_eee(N), N_eeo(N), N_eoe(N), N_eoo(N), N_oee(N), N_oeo(N), N_ooe(N), N_ooo(N), T}(
#         [make_advection_matrix(T, moments[eo_to], moments[switch_eo(eo_to, X)], X) for eo_to in all_eos()]...
#     )
# end

# function advection_matrix(::PNProblem{T, N}, ::Type{Y}) where {T, N}
#     moments = make_moments(N)
#     return SAdvectionMatrixY{N_eee(N), N_eeo(N), N_eoe(N), N_eoo(N), N_oee(N), N_oeo(N), N_ooe(N), N_ooo(N), T}(
#         [make_advection_matrix(T, moments[eo_to], moments[switch_eo(eo_to, Y)], Y) for eo_to in all_eos()]...
#     )
# end

# function advection_matrix(::PNProblem{T, N}, ::Type{Z}) where {T, N}
#     moments = make_moments(N)
#     return SAdvectionMatrixZ{N_eee(N), N_eeo(N), N_eoe(N), N_eoo(N), N_oee(N), N_oeo(N), N_ooe(N), N_ooo(N), T}(
#         [make_advection_matrix(T, moments[eo_to], moments[switch_eo(eo_to, Z)], Z) for eo_to in all_eos()]...
#     )
# end

# function make_zero_advection_matrix(::Type{T}, moments::Moments, dim::Type{<:Dimension}) where T
#     return AdvectionMatrix{SparseMatrixCSC{T}}(
#         [spzeros(length(moments[eo_to]), length(moments[switch_eo(eo_to, dim)])) for eo_to in all_eos()]...
#     )
# end

# function make_advection_matrix_adjoint(::Type{T}, moments::Moments, dim::Type{<:Dimension}) where T
#     return AdvectionMatrix{SparseMatrixCSC{T}}(
#         [-make_advection_matrix(T, moments[eo_to], moments[switch_eo(eo_to, dim)], dim) for eo_to in all_eos()]...
#     )
# end

## BOUNDARY MATRICES

function C(l::Int, k::Int)::Float64
    C = sqrt((2*l + 1)/(2*π) * factorial(l - abs(k))/factorial(l + abs(k)))
    if iseven(abs(k))
        return C
    else
        return -C
    end
end

function doublefactorial(n::Integer)
    if n >= 0
        return Combinatorics.doublefactorial(n)
    else
        return 1
    end
end

function Base.factorial(n::Int64)
    return Base.factorial(big(n))
end

function get_boundary_coefficient(m_to, m_from, dim::Type{X})::Float64
    l, k = m_to
    l´, k´ = m_from
    sum_jq = 0.
    for j in 0:floor(Int, (l - abs(k))/2)
        for q in 0:floor(Int, (l´ - abs(k´))/2) # abs(k´) the ´ is missing
            sum_is = 0.
            for i = 0:floor(Int, (abs(k) - 1)/2)
                for s = 0:floor(Int, (abs(k´) - 1)/2)
                    if k >= 0 && k´ >= 0 && isodd(k) && isodd(k´)
                        sum_is += (-1)^(s + i)*binomial(k, 2*i)*binomial(k´, 2*s)*gamma(0.5 + s + i)*gamma((k + k´)/2 - (s + i))
                    elseif k < 0 && k´ < 0 && iseven(k) && iseven(k´)
                        sum_is += (-1)^(s + i)*binomial(abs(k), 2*i + 1)*binomial(abs(k´), 2*s + 1)*gamma(3/2 + s + i)*gamma((abs(k) + abs(k´))/2 - (s + i) - 1)
                    end
                end
            end
            sum_jq += sum_is * (-1)^(q + j) / ( 2^(j + q)*factorial(j)*factorial(q) ) *
                doublefactorial(2*(l-j)-1) / factorial(l - abs(k) - 2*j) * # factorial in denominator
                doublefactorial(2*(l´-q)-1) / factorial(l´ - abs(k´) - 2*q) * # factorial in denominator
                gamma((l + l´ + 1 - 2*(q + j) - (abs(k) + abs(k´)))/2) / gamma((l + l´)/2 + 1 - (q + j))
        end
    end
    return C(l, abs(k))*C(l´, abs(k´))*(1 + (-1)^(l + l´))*(-1)^(abs(k) + abs(k´))*sum_jq
end

function get_boundary_coefficient(m_to, m_from, dim::Type{Y})::Float64
    l, k = m_to
    l´, k´ = m_from
    sum_jq = 0.
    for j in 0:floor(Int, (l - abs(k))/2)
        for q in 0:floor(Int, (l´ - abs(k´))/2) # abs(k´) the ´ is missing
            sum_is = 0.
            for i = 0:floor(Int, (abs(k) - 1)/2)
                for s = 0:floor(Int, (abs(k´) - 1)/2)
                    sum_is += (-1)^(s + i)*factorial(s + i)*binomial(abs(k), 2*i + 1)*binomial(abs(k´), 2*s + 1)*gamma((abs(k) + abs(k´))/2 - s - i - 0.5)
                end
            end
            sum_jq += sum_is*(-1)^(q + j)/(2^(j + q)*factorial(j)*factorial(q)) *
                doublefactorial(2*(l-j)-1) / factorial(l - abs(k) - 2*j) * # factorial in denominator
                doublefactorial(2*(l´-q)-1) / factorial(l´ - abs(k´) - 2*q) * # factorial in denominator
                gamma((l + l´ + 1 - 2*(q + j) - (abs(k) + abs(k´)))/2) / gamma((l + l´)/2 + 1 - (q + j))
        end
    end
    return C(l, abs(k))*C(l´, abs(k´))*(1 + (-1)^(l + l´))*(-1)^(abs(k) + abs(k´))*sum_jq
end

function get_boundary_coefficient(m_to, m_from, dim::Type{Z})::Float64
    l, k = m_to
    l´, k´ = m_from
    if k == k´
        sum_jq = 0.
        for j in 0:floor(Int, (l - abs(k))/2)
            for q in 0:floor(Int, (l´ - abs(k´))/2)
                sum_jq += 1/(-2)^(q + j) * 1/(factorial(q)*factorial(j)) *
                    doublefactorial(2*(l - j) - 1) / factorial(l - abs(k) - 2*j) *
                    doublefactorial(2*(l´ - q) - 1) / factorial(l´ - abs(k´) - 2*q) * # k -> k´
                    gamma((l + l´)/2 - q - j - abs(k)) / gamma((l + l´)/2 + 1 - q - j)
            end
        end
        return sum_jq*π*C(l, abs(k))*C(l´, abs(k´))*factorial(abs(k))
    else
        return 0.
    end
end

function make_boundary_matrix(::Type{T}, moments_to::Array{Moment, 1}, moments_from::Array{Moment, 1}, dim::Type{<:Dimension}) where T
    A = make_advection_matrix(T, moments_to, moments_from, dim)
    L = spzeros(T, length(moments_to), length(moments_to))
    for (i, m_1) in enumerate(moments_to), (j, m_2) in enumerate(moments_to)
        L[i, j] = get_boundary_coefficient(m_1, m_2, dim)
    end
    return Array(L*A)
end

function make_boundary_matrix_adjoint(::Type{T}, moments_to::Vector{Moment}, moments_from::Vector{Moment}, dim::Type{<:Dimension}) where T
    A = make_advection_matrix(T, moments_to, moments_from, dim)
    L = spzeros(T, length(moments_to), length(moments_to))
    for (i, m_1) in enumerate(moments_to), (j, m_2) in enumerate(moments_to)
        L[i, j] = get_boundary_coefficient(m_1, m_2, dim)
    end
    return Array(-transpose(L)*A)
end

# function make_boundary_matrix(::Type{T}, moments::Moments, ::Type{X}) where T
#     return BoundaryMatrixX(
#         [make_boundary_matrix(T, moments[eo_to], moments[switch_eo(eo_to, X)], X) for eo_to in all_eos() if is_odd_in(eo_to, X)]...
#     )
# end

# function make_boundary_matrix(::Type{T}, moments::Moments, ::Type{Y}) where T
#     return BoundaryMatrixY(
#         [make_boundary_matrix(T, moments[eo_to], moments[switch_eo(eo_to, Y)], Y) for eo_to in all_eos() if is_odd_in(eo_to, Y)]...
#     )
# end

# function make_boundary_matrix(::Type{T}, moments::Moments, ::Type{Z}) where T
#     return BoundaryMatrixZ(
#         [make_boundary_matrix(T, moments[eo_to], moments[switch_eo(eo_to, Z)], Z) for eo_to in all_eos() if is_odd_in(eo_to, Z)]...
#     )
# end

function make_boundary_matrix_adjoint(::Type{T}, moments::Moments, ::Type{X}) where T
    return BoundaryMatrixX(
        [make_boundary_matrix_adjoint(T, moments[eo_to], moments[switch_eo(eo_to, X)], X) for eo_to in all_eos() if is_odd_in(eo_to, X)]...
    )
end

function make_boundary_matrix_adjoint(::Type{T}, moments::Moments, ::Type{Y}) where T
    return BoundaryMatrixY(
        [make_boundary_matrix_adjoint(T, moments[eo_to], moments[switch_eo(eo_to, Y)], Y) for eo_to in all_eos() if is_odd_in(eo_to, Y)]...
    )
end

function make_boundary_matrix_adjoint(::Type{T}, moments::Moments, ::Type{Z}) where T
    return BoundaryMatrixZ(
        [make_boundary_matrix_adjoint(T, moments[eo_to], moments[switch_eo(eo_to, Z)], Z) for eo_to in all_eos() if is_odd_in(eo_to, Z)]...
    )
end

function beam_moments(moments::Array{Moment, 1}, beam_direction::Distribution)
    Nmax = maximum(degree.(moments))
    f((θ, ϕ)) = -SphericalHarmonics.computeYlm(θ, ϕ; lmax=Nmax, SHType=SphericalHarmonics.RealHarmonics())*
                    pdf(beam_direction, [sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), cos(θ)])*
                    sin(θ)
    moms, error = hcubature(f, [0., 0.], [π, 2*π])
    return [moms[(m..., )] for m in moments]
end
