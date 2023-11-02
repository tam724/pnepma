include("../pn_solver/pn_solver.jl")

##
using Optim
using Serialization

## PROBLEM SETUP
DTYPE = Float64
elms = @SVector [n"Ni", n"Fe"]
x_rays = @SVector [n"Ni K-L2", n"Ni K-L3", n"Fe K-L2", n"Fe K-L3"]

## compute electron fluences for multiple energies
m_std = VolumeFractionPNMaterial(elms, ConstPNParametrization{2}([0.5, 0.5]))
M = (100, 1, 1)
gspec = make_gridspec(
    M,
    DTYPE.(convert_strip_length.((-500.0u"nm", 0.0u"nm"))),
    DTYPE.(convert_strip_length.(0u"nm")),
    DTYPE.(convert_strip_length.(0u"nm")))

detector_position = convert_strip_length.([10.0u"nm", 0.0u"nm", 0.0u"nm"])

beam_energies = [convert_strip_energy(e * 1u"keV") for e in range(9.0, 15.0, length=5)]
beam_energies_test = [convert_strip_energy(e * 1u"keV") for e in [11.0, 14.0]]

make_problem(e) = ForwardPNProblem{9, Float64}(
        gspec,
        m_std.elms,
        #beam_energy
        Normal(
            e,
            convert_strip_energy(0.1u"keV")),
        #beam_position
        MultivariateNormal(
            convert_strip_length.([0.0u"nm", 0.0u"nm", 0.0u"nm"]),
            convert_strip_length(30u"nm")),
        #beam_direction
        VonMisesFisher(
            [-1., 0., 0.], 10.
        ),
        #E_initial
        DTYPE(convert_strip_energy(15.5u"keV")),
        #E_cutoff
        DTYPE(convert_strip_energy(7.0u"keV")),
        #N_steps
        1000)

problems = [make_problem(e) for e in beam_energies]
problems_test = [make_problem(e) for e in beam_energies_test]

##
std_ints = [calc_intensities(p, m_std, x_rays, detector_position) for p in problems]
std_ints_test = [calc_intensities(p, m_std, x_rays, detector_position) for p in problems_test]
##
function measure(m)
    return vcat([calc_intensities(problems[i], m, x_rays, detector_position) ./ std_ints[i] for i in 1:length(problems)]...)
end

function measure_test(m)
    return vcat([calc_intensities(problems[i], m, x_rays, detector_position) ./ std_ints_test[i] for i in 1:length(problems_test)]...)
end

function l2_measure(m)
    ρ = component_densities_eee(m, gspec)
    return vcat(ρ...)
end

function loss(m, k_exp)
    k = measure(m)
    return sum((k - k_exp).^2)
end

function loss_test(m, k_exp_test)
    k = measure_test(m)
    return sum((k - k_exp_test).^2)
end

function l2_loss(m, l2_exp)
    ρ = l2_measure(m)
    return sum((l2_exp - ρ).^2)    
end


function ref_func(x)
    if x[1] < convert_strip_length(-60.0u"nm")
        return [1.0]
    else
        return [0.0]
    end
end

m_ref = VolumeFractionPNMaterialSC(elms,
    FunctionPNParametrization{1}(x -> ref_func(x))
)

k_exp = measure(m_ref)
k_exp_test = measure_test(m_ref)
l2_exp = l2_measure(m_ref)

##

m_opti_box = VolumeFractionPNMaterialSC(elms,
    fill(BoxedPNParametrization{false, 1}, [1.0],
        convert_strip_length.((-300.0u"nm", 0.0u"nm")),
        convert_strip_length.((-100.0u"nm", 100.0u"nm")),
        convert_strip_length.((-100.0u"nm", 100.0u"nm")),
        10, 1, 1, 0.5)
)

m_opti_bil = VolumeFractionPNMaterialSC(elms,
    fill(BilinearPNParametrization{false, 1}, [1.0],
        convert_strip_length.((-300.0u"nm", 0.0u"nm")),
        convert_strip_length.((0.0u"nm", 0.0u"nm")),
        convert_strip_length.((0.0u"nm", 0.0u"nm")),
        10, 1, 1, 0.5)
)

m_opti_nn = VolumeFractionPNMaterialSC(elms,
    NNPNParametrization{1, 1}(
        Chain(
            Normalization(convert_strip_length.((-300.0u"nm", 0.0u"nm"))),
            Dense(1, 3, Tanh()),
            Dense(3, 1, Sigmoid())
            )
    )
)

m_opti = m_opti_bil
m_opti_name = "bil"
m_opti_grad = zero(m_opti)

##

function loss_fg!(F, G, x) 
    # from_param_vec!(m_opti, clamp.(x, 0., 1.))
    from_param_vec!(m_opti, x)
    display(plot(m_opti, gspec, 1))
    if G !== nothing
        temp = Zygote.withgradient(m_ -> loss(m_, k_exp), m_opti)
        loss_ = temp.val
        (m_grad, ) = temp.grad
        from_named_tuple!(m_opti_grad, m_grad)
        param_vec!(G, m_opti_grad)
        return loss_
    end
    if F !== nothing
        loss_ = loss(m_opti, k_exp)
        return loss_
    end
end

my_loss_test = []
my_l2_loss = []

function callback(x)
    @show x[end].metadata["x"]
    from_param_vec!(m_opti, x[end].metadata["x"])
    L_test = loss_test(m_opti, k_exp_test)
    L_L2 = l2_loss(m_opti, l2_exp)
    push!(my_loss_test, L_test)
    push!(my_l2_loss, L_L2)
    return false
end

p = param_vec(m_opti)
optimizer = Fminbox(BFGS())

##
opt = optimize(Optim.only_fg!(loss_fg!), zeros(n_params(m_opti)), ones(n_params(m_opti)), p, optimizer, Optim.Options(store_trace=true, callback=callback, show_trace=true, extended_trace=true, iterations=3000))
# opt = optimize(Optim.only_fg!(loss_fg!), p, BFGS(), Optim.Options(store_trace=true, callback=callback, show_trace=true, extended_trace=true, iterations=3000))

serialize(string(m_opti_name, ".jls"), (opt, my_loss_test, my_l2_loss))