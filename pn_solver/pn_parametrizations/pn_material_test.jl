using Test
using FiniteDifferences

include("../../StaRMAP/StaRMAP.jl")
using .StaRMAP
include("../units.jl")
using NeXLCore
include("pn_material.jl")

function test_function(m::PNMaterial, x)
    return sum(component_densities(m, x))
end

##
function compute_jacobian_forward_diff(m::PNMaterial, x)
    ForwardDiff_gradient(m->test_function(m, x), m)'
end

function compute_jacobian_finite_differences(m::PNMaterial, x)
    new_m = zero(m)
    p = zeros(n_params(new_m))
    p = param_vec!(p, m)
    p = deepcopy(p)
    from_param_vec!(new_m, p)

    return FiniteDifferences.jacobian(central_fdm(10, 1; adapt=10), p -> test_function(from_param_vec!(new_m, p), x), p)[1]
end
##
function compute_jacobian_backprop(m::PNMaterial, x)
    jacobian_backprop = zeros(1, n_params(m))
    m_ = zero(m)
    ρ = zeros(length(m.elms))
    ρ_ = ones(length(m.elms))
    component_densities_pullback!(ρ, m, x, ρ_, m_)
    param_vec!(@view(jacobian_backprop[1, :]), m_)
    return jacobian_backprop
end

function compare_backprop_with_FD_gradient(m::PNMaterial, val=1e-7)
    x = rand(3)
    @show x

    p = zeros(n_params(m))
    p = param_vec!(p, m)

    jacobian_finite_differences = compute_jacobian_finite_differences(m, x)
    jacobian_backprop = compute_jacobian_backprop(m, x)
    # @show typeof(jacobian_backprop)
    # @show size(jacobian_backprop)
    # @show jacobian_finite_differences

    # p = Plots.plot(jacobian_backprop[:])
    # Plots.plot!(jacobian_finite_differences[:])
    # display(p)

    error = jacobian_backprop .- jacobian_finite_differences
    @show maximum(abs.(error))
    return all(isapproxzero.(error; val=val))
end

function compare_backprop_with_FORD_gradient(m::PNMaterial, val=1e-7)
    x = rand(3)
    @show x

    p = zeros(n_params(m))
    p = param_vec!(p, m)
    p_copy = deepcopy(p)

    jacobian_forward_diff = compute_jacobian_forward_diff(m, x)
    from_param_vec!(m, p_copy)
    jacobian_backprop = compute_jacobian_backprop(m, x)

    error = jacobian_backprop .- jacobian_forward_diff
    @show maximum(abs.(error))
    return all(isapproxzero.(error; val=val))
end

##
els = [n"Cu", n"Ni"]
els3 = [n"Cu", n"Ni", n"Si"]

parametrizations = [
    ConstPNParametrization{length(els)}([0.4, 0.2]),
    BilinearPNParametrization{true, length(els)}([rand(length(els)) for _ in 1:5, _ in 1:5, _ in 1:5], [0.4, 0.6], range(0.25, 0.75, length=5), range(0.25, 0.75, length=5), range(0.25, 0.75, length=5)),
    BilinearPNParametrization{false, length(els)}([rand(length(els)) for _ in 1:5, _ in 1:5, _ in 1:5], [0.4, 0.6], range(0.25, 0.75, length=5), range(0.25, 0.75, length=5), range(0.25, 0.75, length=5)),
    BoxedPNParametrization{true, length(els)}([rand(2) for _ in 1:4, _ in 1:4, _ in 1:4], [0.4, 0.6], range(0., 0.5, length=5), range(0., 0.5, length=5), range(0., 0.5, length=5)),
    BoxedPNParametrization{false, length(els)}([rand(2) for _ in 1:4, _ in 1:4, _ in 1:4], [0.4, 0.6], range(0., 0.5, length=5), range(0., 0.5, length=5), range(0., 0.5, length=5)),
    NNPNParametrization{length(els)}(Chain(Dense(3, 5, Tanh()), Dense(5, length(els), Id()), Softmax(length(els)))),
    NNPNParametrization{length(els), 3}(Chain(GaussianDistance(3, 3), Dense(3, length(els), Id()), Softmax(length(els)))),
    NNPNParametrization{length(els), 2}(
        Chain(
            EllipseLayer(
                [0.1, 0.2],
                [1., 2.],
                [0.4]
            ),
            Dense(1, length(els), Sigmoid())
        )
    )
    ]

##
##
for p in parametrizations
    m = MassConcentrationPNMaterial(els, p)
    @show typeof(p)
    @show @test compare_backprop_with_FD_gradient(m)
    if typeof(m.p) <: ConstPNParametrization || typeof(m.p) <: BoxedPNParametrization || typeof(m.p) <: BilinearPNParametrization
        @show @test compare_backprop_with_FORD_gradient(m)
    end
end

##

for p in parametrizations
    m = MassFractionPNMaterial(els, p)
    mSC = MassFractionPNMaterialSC(els3, p)
    @show typeof(p)
    @show @test compare_backprop_with_FD_gradient(m)
    @show @test compare_backprop_with_FD_gradient(mSC, 1e-6)
    if typeof(m.p) <: ConstPNParametrization || typeof(m.p) <: BoxedPNParametrization || typeof(m.p) <: BilinearPNParametrization
        @show @test compare_backprop_with_FORD_gradient(m)
        @show @test compare_backprop_with_FORD_gradient(mSC)
    end
end

##

for p in parametrizations
    m = VolumeFractionPNMaterial(els, p)
    mSC = VolumeFractionPNMaterialSC(els3, p)
    @show typeof(p)
    @show @test compare_backprop_with_FD_gradient(m)
    @show @test compare_backprop_with_FD_gradient(mSC)
    if typeof(m.p) <: ConstPNParametrization || typeof(m.p) <: BoxedPNParametrization || typeof(m.p) <: BilinearPNParametrization
        @show @test compare_backprop_with_FORD_gradient(m)
        @show @test compare_backprop_with_FORD_gradient(mSC)
    end
end

##

for p in parametrizations
    m = LinearDensityPNMaterial(els, p)
    mSC = LinearDensityPNMaterialSC(els3, p)
    @show typeof(p)
    @show @test compare_backprop_with_FD_gradient(m, 1e-6)
    @show @test compare_backprop_with_FD_gradient(mSC, 1e-6)
    if typeof(m.p) <: ConstPNParametrization || typeof(m.p) <: BoxedPNParametrization || typeof(m.p) <: BilinearPNParametrization
        @show @test compare_backprop_with_FORD_gradient(m)
        @show @test compare_backprop_with_FORD_gradient(mSC)
    end
end

##