abstract type ActivationFunction end
struct PNAdjoint{T} end

adjoint(::T) where {T <: ActivationFunction} = PNAdjoint{T}()

##ACTIVATION FUNCTIONS
struct Tanh <: ActivationFunction end
(::Tanh)(x) = tanh(x)
(::PNAdjoint{Tanh})(x) = 1. - tanh(x)^2 #2. /(cosh(2. * x) + 1.)

struct ReLU <: ActivationFunction end
(::ReLU)(x) = max(0., x)
(::PNAdjoint{ReLU})(x) = x > 0 ? 1.0 : 0.0

struct Id <: ActivationFunction end
(::Id)(x) = x
(::PNAdjoint{Id})(x) = 1.0

struct Sigmoid <: ActivationFunction end
(::Sigmoid)(x) = NNlib.sigmoid(x)
function (::PNAdjoint{Sigmoid})(x)
    σ = NNlib.sigmoid(x)
    return σ*(1. - σ)
end

struct LeakyReLU{P} <: ActivationFunction end
(::LeakyReLU{P})(x) where P = max(P*x, x)
(::PNAdjoint{LeakyReLU{P}})(x) where P = x > 0 ? 1.0 : P ## assert P < 1.

struct Softplus <: ActivationFunction end
(::Softplus)(x) = NNlib.softplus(x)
(::PNAdjoint{Softplus})(x) = 1. / (1. + exp(-x))

abstract type Layer end

#### DENSE LAYER
struct Dense{F<:ActivationFunction, M, B, O} <: Layer
    weight::M   
    bias::B
    σ::F
    z::O
    output::O
end

Base.zero(a::Dense) = Dense(zero(a.weight), zero(a.bias), deepcopy(a.σ), zero(a.z), zero(a.output))
n_params(a::Dense) = length(a.weight) + length(a.bias)
n_inputs(a::Dense) = size(a.weight, 2)
n_outputs(a::Dense) = size(a.weight, 1)
Flux.trainable(a::Dense) = (a.weight, a.bias)

to_named_tuple(a::Dense) = (weight=a.weight, bias=a.bias)
function from_named_tuple!(a::Dense, p::NamedTuple)
    a.weight .= p.weight
    a.bias .= p.bias
end

function param_vec!(p::AbstractVector, a::Dense)
    p[1:length(a.weight)] .= a.weight[:]
    p[length(a.weight)+1:length(a.weight) + length(a.bias)] .= a.bias
    return p
end

function from_param_vec!(a::Dense, p::AbstractVector)
    a.weight .= reshape(p[1:length(a.weight)], size(a.weight))
    a.bias .= p[length(a.weight)+1 : length(a.weight) + length(a.bias)]
    return a
end

function Dense(in, out, σ)
    W = Flux.glorot_uniform(out, in)

    return Dense(Float64.(W), zeros(out), σ, zeros(out), zeros(out))
    # return Dense(rand(out, in), zeros(out), σ, zeros(out), zeros(out))
end

function feed!(a::Dense, x::AbstractVector)
    # a.z .= a.weight*x .+ a.bias -> INPLACE VERSION
    mul!(a.z, a.weight, x)
    a.z .+= a.bias
    a.output .= a.σ.(a.z)
end

function feed_pullback!(a::Dense, x::AbstractVector, a_::Dense, x_::AbstractVector)
    a_.z .= a.σ'.(a.z) .* a_.output
    a_.weight .+= a_.z .* x'
    a_.bias .+= a_.z
    mul!(x_, a.weight', a_.z)
end

function feed_pullback!(a::Dense, x::AbstractVector, a_::Dense, ::Nothing)
    a_.z .= a.σ'.(a.z) .* a_.output
    a_.weight .+= a_.z .* x'
    a_.bias .+= a_.z
    # x_ .= a.weight' * a_.z
end

#### DENSE LAYER WITHOUT BIAS
struct DenseNoBias{F<:ActivationFunction, M, O} <: Layer
    weight::M   
    σ::F
    z::O
    output::O
end

Base.zero(a::DenseNoBias) = DenseNoBias(zero(a.weight), deepcopy(a.σ), zero(a.z), zero(a.output))
n_params(a::DenseNoBias) = length(a.weight)
n_inputs(a::DenseNoBias) = size(a.weight, 2)
n_outputs(a::DenseNoBias) = size(a.weight, 1)
Flux.trainable(a::DenseNoBias) = (a.weight,)

to_named_tuple(a::DenseNoBias) = (weight=a.weight,)
function from_named_tuple!(a::DenseNoBias, p::NamedTuple)
    a.weight .= p.weight
end

function param_vec!(p::AbstractVector, a::DenseNoBias)
    p[1:length(a.weight)] .= a.weight[:]
    return p
end

function from_param_vec!(a::DenseNoBias, p::AbstractVector)
    a.weight .= reshape(p[1:length(a.weight)], size(a.weight))
    return a
end

function DenseNoBias(in, out, σ)
    W = Flux.glorot_uniform(out, in)
    return DenseNoBias(Float64.(W), σ, zeros(out), zeros(out))
end

function feed!(a::DenseNoBias, x::AbstractVector)
    # a.z .= a.weight*x .+ a.bias -> INPLACE VERSION
    mul!(a.z, a.weight, x)
    a.output .= a.σ.(a.z)
end

function feed_pullback!(a::DenseNoBias, x::AbstractVector, a_::DenseNoBias, x_::AbstractVector)
    a_.z .= a.σ'.(a.z) .* a_.output
    a_.weight .+= a_.z .* x'
    mul!(x_, a.weight', a_.z)
end

function feed_pullback!(a::DenseNoBias, x::AbstractVector, a_::DenseNoBias, ::Nothing)
    a_.z .= a.σ'.(a.z) .* a_.output
    a_.weight .+= a_.z .* x'
    # x_ .= a.weight' * a_.z
end

### NormalizationLayer
struct Normalization{W, B, O} <: Layer
    weight::W
    bias::B
    output::O
end

Base.zero(a::Normalization) = Normalization(deepcopy(a.weight), deepcopy(a.bias), zero(a.output))
n_params(::Normalization) = 0
n_inputs(a::Normalization) = size(a.weight, 2)
n_outputs(a::Normalization) = size(a.weights, 1)
Flux.trainable(::Normalization) = nothing

to_named_tuple(::Normalization) = ()
from_named_tuple!(::Normalization, ::NamedTuple) = nothing
from_named_tuple!(::Normalization, ::Tuple{}) = nothing

param_vec!(::AbstractVector, ::Normalization) = nothing
from_param_vec!(::Normalization, ::AbstractVector) = nothing

function Normalization(x::Tuple, y::Tuple, z::Tuple)
    # normalizes the input of the layer w.r.t. to the range x, y and z
    Normalization(Diagonal([2. /(x[2] - x[1]), 2. /(y[2] - y[1]), 2. /(z[2] - z[1])]), [(x[2] + x[1])/(x[1] - x[2]), (y[2] + y[1])/(y[1] - y[2]), (z[2] + z[1])/(z[1] - z[2])], zeros(3))
end

function Normalization(x::Tuple, y::Tuple)
    # normalizes the input of the layer w.r.t. to the range x, y and z
    Normalization(Diagonal([2. /(x[2] - x[1]), 2. /(y[2] - y[1])]), [(x[2] + x[1])/(x[1] - x[2]), (y[2] + y[1])/(y[1] - y[2])], zeros(2))
end

function Normalization(x::Tuple)
    # normalizes the input of the layer w.r.t. to the range x, y and z
    Normalization(Diagonal([2. /(x[2] - x[1])]), [(x[2] + x[1])/(x[1] - x[2])], zeros(1))
end

function feed!(a::Normalization, x::AbstractVector)
    # a.z .= a.weight*x -> INPLACE VERSION
    mul!(a.output, a.weight, x)
    a.output .+= a.bias
end

function feed_pullback!(a::Normalization, _::AbstractVector, a_::Normalization, x_::AbstractVector)
    mul!(x_, a.weight', a_.output)
end

function feed_pullback!(_::Normalization, _::AbstractVector, _::Normalization, ::Nothing)
    nothing
end

### SOFTMAX LAYER
struct Softmax{O<:Vector} <: Layer
    output::O
end

Softmax(n::Integer) = Softmax(zeros(n))

Base.zero(a::Softmax) = Softmax(zero(a.output))
n_params(a::Softmax) = 0
n_inputs(a::Softmax) = length(a.output)
n_output(a::Softmax) = length(a.output)
Flux.trainable(::Softmax) = nothing

to_named_tuple(::Softmax) = ()
from_named_tuple!(::Softmax, ::NamedTuple) = nothing
from_named_tuple!(::Softmax, ::Tuple{}) = nothing

param_vec!(::AbstractVector, ::Softmax) = nothing
from_param_vec!(::Softmax, ::AbstractVector) = nothing

function feed!(a::Softmax, x::AbstractVector)
    NNlib.softmax!(a.output, x) # using a stable softmax implementation
end

function feed_pullback!(a::Softmax, x::AbstractVector, a_::Softmax, x_::AbstractVector)
    temp = dot(a.output, a_.output)
    x_ .= a.output .* (a_.output .- temp)
end

## GAUSSIAN DISTANCE LAYER
struct GaussianDistance{M, V, O} <: Layer
    means::M # vector/vector
    vars::V #vector matrix
    dists::O
    output::O
end

rand_m1p1() = 2. * rand() - 1.
GaussianDistance(in::Integer, out::Integer) = GaussianDistance([[rand_m1p1() for _ in 1:in] for _ in 1:out], [[rand_m1p1() for _ in 1:in, _ in 1:in] for _ in 1:out], zeros(out), zeros(out))

function Base.zero(a::GaussianDistance)
    in = n_inputs(a)
    out = n_outputs(a)
    return GaussianDistance([zeros(in) for _ in 1:out], [zeros(in, in) for _ in 1:out], zeros(out), zeros(out))
end
n_params(a::GaussianDistance) = (n_inputs(a) + n_inputs(a)*n_inputs(a))*n_outputs(a)
n_inputs(a::GaussianDistance) = length(a.means[1])
n_outputs(a::GaussianDistance) = length(a.output)
Flux.trainable(a::GaussianDistance) = (a.means..., a.vars...)

to_named_tuple(a::GaussianDistance) = (means=a.means, vars=a.vars)
function from_named_tuple!(a::GaussianDistance, p::NamedTuple)
    for i in 1:n_outputs(a)
        a.means[i] .= p.means[i]
        a.vars[i] .= p.vars[i]
    end
end

function param_vec!(p::AbstractVector, a::GaussianDistance)
    p_per_blob = (n_inputs(a) + n_inputs(a)*n_inputs(a))
    for i in 1:n_outputs(a)
        p[(i-1)*p_per_blob+1:(i-1)*p_per_blob + n_inputs(a)] .= a.means[i][:]
        p[(i-1)*p_per_blob + n_inputs(a) + 1: (i-1)*p_per_blob + n_inputs(a) + n_inputs(a)*n_inputs(a)] .= a.vars[i][:]
    end
end
        
function from_param_vec!(a::GaussianDistance, p::AbstractVector)
    p_per_blob = (n_inputs(a) + n_inputs(a)*n_inputs(a))
    for i in 1:n_outputs(a)
        a.means[i][:] .= p[(i-1)*p_per_blob+1:(i-1)*p_per_blob + n_inputs(a)]
        a.vars[i][:] .= p[(i-1)*p_per_blob + n_inputs(a) + 1: (i-1)*p_per_blob + n_inputs(a) + n_inputs(a)*n_inputs(a)]
    end
end

function feed!(a::GaussianDistance, x::AbstractVector)
    # compute distances
    for i in 1:n_outputs(a)
        a.dists[i] = (x .- a.means[i])' * a.vars[i] * (x .- a.means[i])
    end
    a.output .= exp.(.-a.dists)
end

function feed_pullback!(a::GaussianDistance, x::AbstractVector, a_::GaussianDistance, x_::AbstractVector)
    a_.dists .= .- a.output .* a_.output
    x_ .= 0.
    for i in 1:n_outputs(a)
        x_ .+= ((a.vars[i] .+ a.vars[i]') * (x .- a.means[i])) * a_.dists[i]
        a_.means[i] .= - ((a.vars[i] .+ a.vars[i]') * (x .- a.means[i])) .* a_.dists[i]
        a_.vars[i] .= (x .- a.means[i]) * (x .- a.means[i])' .* a_.dists[i]
    end
end

function feed_pullback!(a::GaussianDistance, x::AbstractVector, a_::GaussianDistance, ::Nothing)
    a_.dists .= .- a.output .* a_.output
    for i in 1:n_outputs(a)
        a_.means[i] .= - ((a.vars[i] .+ a.vars[i]') * (x .- a.means[i])) .* a_.dists[i]
        a_.vars[i] .= (x .- a.means[i]) * (x .- a.means[i])' .* a_.dists[i]
    end
end

## SINGLE DISTANCE LAYER
struct SingleDistance{M, S, O} <: Layer
    mean::M
    scale::S
    output::O
end

SingleDistance(mean::M, scale::S) where {M, S} = SingleDistance(mean, scale, zeros(1))

Base.zero(a::SingleDistance) = SingleDistance(deepcopy(a.mean), deepcopy(a.scale), zeros(1))
n_params(a::SingleDistance) = 0
n_inputs(a::SingleDistance) = length(a.mean)
n_outputs(a::SingleDistance) = 1
Flux.trainable(::SingleDistance) = nothing

to_named_tuple(::SingleDistance) = ()
from_named_tuple!(::SingleDistance, ::NamedTuple) = nothing
from_named_tuple!(::SingleDistance, ::Tuple{}) = nothing

param_vec!(::AbstractVector, ::SingleDistance) = nothing
from_param_vec!(::SingleDistance, ::AbstractVector) = nothing

function feed!(a::SingleDistance, x::AbstractVector)
    a.output[1] = norm(x .- a.mean) / a.scale
end

function feed_pullback!(a::SingleDistance, x::AbstractVector, a_::SingleDistance, x_::AbstractVector)
end

function feed_pullback!(a::SingleDistance, x::AbstractVector, a_::SingleDistance, ::Nothing)
end

## SINGLE DISTANCE LAYER
struct EllipseLayer{M, S, R, O} <: Layer
    mean::M
    scale::S
    rot::R
    output::O
end

EllipseLayer(mean::M, scale::S, rot::R) where {M, S, R} = EllipseLayer(mean, scale, rot, zeros(1))

Base.zero(a::EllipseLayer) = EllipseLayer(zero(a.mean), zero(a.scale), zero(a.rot))
n_params(a::EllipseLayer) = 5
n_inputs(a::EllipseLayer) = 2
n_outputs(a::EllipseLayer) = 1
Flux.trainable(a::EllipseLayer) = (a.mean, a.scale, a.rot)

to_named_tuple(a::EllipseLayer) = (mean=a.mean, scale=a.scale, rot=a.rot)
function from_named_tuple!(a::EllipseLayer, p::NamedTuple)
    a.mean .= p.mean
    a.scale .= p.scale
    a.rot .= p.rot
end

function param_vec!(p::AbstractVector, a::EllipseLayer)
    p[1:2] .= a.mean
    p[3:4] .= a.scale
    p[5:5] .= a.rot
end

function from_param_vec!(a::EllipseLayer, p::AbstractVector)
    a.mean .= p[1:2]
    a.scale .= p[3:4]
    a.rot .= p[5:5]
end

function feed!(a::EllipseLayer, x::AbstractVector)
    xx = x[1] - a.mean[1]
    yy = x[2] - a.mean[2]
    a.output[1] = (xx*cos(a.rot[1]) + yy*sin(a.rot[1]))^2 / a.scale[1]^2 + (xx*sin(a.rot[1]) - yy*cos(a.rot[1]))^2 / a.scale[2]^2
end

function feed_pullback!(a::EllipseLayer, x::AbstractVector, a_::EllipseLayer, x_::AbstractVector)
    xx = x[1] - a.mean[1]
    yy = x[2] - a.mean[2]
    xx_ = (2*cos(a.rot[1])*(xx*cos(a.rot[1]) + yy*sin(a.rot[1]))/a.scale[1]^2 + 2*sin(a.rot[1])*(xx*sin(a.rot[1]) - yy*cos(a.rot[1]))/a.scale[2]^2)*a_.output[1]
    yy_ = (2*sin(a.rot[1])*(xx*cos(a.rot[1]) + yy*sin(a.rot[1]))/a.scale[1]^2 - 2*cos(a.rot[1])*(xx*sin(a.rot[1]) - yy*cos(a.rot[1]))/a.scale[2]^2)*a_.output[1]
    a_.rot[1] += 2*(a.scale[1]^2 - a.scale[2]^2)*(xx*cos(a.rot[1]) + yy*sin(a.rot[1]))*(xx*sin(a.rot[1]) - yy*cos(a.rot[1]))/(a.scale[1]^2*a.scale[2]^2)*a_.output[1]
    a_.scale[1] -= 2*(xx*cos(a.rot[1]) + yy*sin(a.rot[1]))^2/a.scale[1]^3*a_.output[1]
    a_.scale[2] -= 2*(xx*sin(a.rot[1]) - yy*cos(a.rot[1]))^2/a.scale[2]^3*a_.output[1]
    a_.mean[1] -= xx_
    a_.mean[2] -= yy_
    x_[1] = xx_
    x_[2] = yy_
end

function feed_pullback!(a::EllipseLayer, x::AbstractVector, a_::EllipseLayer, ::Nothing)
    xx = x[1] - a.mean[1]
    yy = x[2] - a.mean[2]
    xx_ = (2*cos(a.rot[1])*(xx*cos(a.rot[1]) + yy*sin(a.rot[1]))/a.scale[1]^2 + 2*sin(a.rot[1])*(xx*sin(a.rot[1]) - yy*cos(a.rot[1]))/a.scale[2]^2)*a_.output[1]
    yy_ = (2*sin(a.rot[1])*(xx*cos(a.rot[1]) + yy*sin(a.rot[1]))/a.scale[1]^2 - 2*cos(a.rot[1])*(xx*sin(a.rot[1]) - yy*cos(a.rot[1]))/a.scale[2]^2)*a_.output[1]
    a_.rot[1] += 2*(a.scale[1]^2 - a.scale[2]^2)*(xx*cos(a.rot[1]) + yy*sin(a.rot[1]))*(xx*sin(a.rot[1]) - yy*cos(a.rot[1]))/(a.scale[1]^2*a.scale[2]^2)*a_.output[1]
    a_.scale[1] -= 2*(xx*cos(a.rot[1]) + yy*sin(a.rot[1]))^2/a.scale[1]^3*a_.output[1]
    a_.scale[2] -= 2*(xx*sin(a.rot[1]) - yy*cos(a.rot[1]))^2/a.scale[2]^3*a_.output[1]
    a_.mean[1] -= xx_
    a_.mean[2] -= yy_
end

### CHAIN DEFINITION
struct Chain{T <: Tuple}
    layers::T
end

Chain(layers...) = Chain(layers)
Chain(layer::Layer) = Chain((layer, ))

Base.zero(c::Chain) = Chain(zero.(c.layers))
n_params(c::Chain) = sum(n_params.(c.layers))
n_inputs(c::Chain) = n_inputs(first(c.layers))
n_outputs(c::Chain) = n_outputs(last(c.layers))

Flux.trainable(c::Chain) = Flux.trainable.(c.layers)

to_named_tuple(c::Chain) = (layers=to_named_tuple.(c.layers), )
from_named_tuple!(c::Chain, p::NamedTuple) = from_named_tuple!.(c.layers, p.layers)

function param_vec!(p::AbstractVector, c::Chain)
    i = 0
    for l in c.layers
        param_vec!(@view(p[i+1:i + n_params(l)]), l)
        i = i + n_params(l)
    end
    return p
end

function from_param_vec!(c::Chain, p::AbstractVector)
    i = 0
    for l in c.layers
        from_param_vec!(l, @view(p[i+1:i + n_params(l)]))
        i = i + n_params(l)
    end
    return c
end

feed!(::Tuple{}, x) = nothing
feed_pullback!(::Tuple{}, x, ::Tuple{}, x_) = nothing

function feed!(fs::Tuple, x)
    feed!(first(fs), x)
    feed!(Base.tail(fs), first(fs).output)
end

function feed_pullback!(fs::Tuple{<:Layer}, x, fs_::Tuple{<:Layer})
    feed_pullback!(last(fs), x, last(fs_), nothing)
end

function feed_pullback!(fs::Tuple, x, fs_::Tuple)
    # the tuples have min 2 elements
    remaining = Base.front(fs)
    remaining_ = Base.front(fs_)
    feed_pullback!(last(fs), last(remaining).output, last(fs_), last(remaining_).output) #  calls pullback for one layer
    feed_pullback!(remaining, x, remaining_) # calls pullback for the tuple{layers...}
end

function feed!(res::AbstractVector, c::Chain, x::AbstractVector)
    feed!(c.layers, x)
    res .= last(c.layers).output
end

function feed!(::Nothing, c::Chain, x::AbstractVector)
    feed!(c.layers, x)
end

function feed_pullback!(res, c::Chain, x, res_::AbstractVector, c_::Chain)
    last(c_.layers).output .= res_
    feed_pullback!(c.layers, x, c_.layers)
end