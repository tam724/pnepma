using Test
using FiniteDifferences

include("pn_material.jl")

function compute_jacobian_finite_differences(c::Chain, x)
    res = zeros(n_outputs(c))
    p = zeros(n_params(c))
    p = param_vec!(p, c)

    return FiniteDifferences.jacobian(central_fdm(2, 1), p -> feed!(res, from_param_vec!(c, p), x), p)[1]
end

function compute_jacobian_backprop(c::Chain, x)
    jacobian_backprop = zeros(n_outputs(c), n_params(c))
    res = zeros(n_outputs(c))
    res_ = zeros(n_outputs(c))
    for i in 1:n_outputs(c)
        res_[i] = 1.0
        c_ = zero(c)
        feed!(res, c, x)
        feed_pullback!(res, c, x, res_, c_)
        param_vec!(@view(jacobian_backprop[i, :]), c_)
        res_[i] = 0.0
    end
    return jacobian_backprop
end

function compare_backprop_with_FD_gradient(c::Chain)
    x = rand(n_inputs(c))

    p = zeros(n_params(c))
    p = param_vec!(p, c)
    p_copy = deepcopy(p)

    jacobian_finite_differences = compute_jacobian_finite_differences(c, x)
    from_param_vec!(c, p_copy)
    jacobian_backprop = compute_jacobian_backprop(c, x)

    return all(isapproxzero.(jacobian_backprop .- jacobian_finite_differences; val=1e-8))
end

##

@test compare_backprop_with_FD_gradient(Chain(Dense(3, 2, Tanh()), Dense(2, 10, Sigmoid()), Softmax(10)))
##
@test compare_backprop_with_FD_gradient(Chain(Dense(3, 10, Tanh()), Dense(10, 2, Tanh()), Dense(2, 2, Tanh())))
##
@test compare_backprop_with_FD_gradient(Chain(Dense(3, 10, ReLU()), Dense(10, 2, Tanh()), Dense(2, 2, Id())))
##
@test compare_backprop_with_FD_gradient(Chain(Dense(3, 10, Id()), Dense(10, 2, ReLU()), Dense(2, 2, ReLU())))
##
@test compare_backprop_with_FD_gradient(Chain(Dense(3, 10, ReLU()), Dense(10, 2, ReLU()), Dense(2, 2, Tanh())))
##
@test compare_backprop_with_FD_gradient(Chain(DenseNoBias(3, 10, ReLU()), Dense(10, 2, ReLU()), DenseNoBias(2, 2, Tanh())))
