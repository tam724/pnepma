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

##
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

##

function ref_func(x)
    a = sigmoid(-700000.0*(x[1] - convert_strip_length(-120.0u"nm")))
    b = 0.4 * exp(-80000000000.0*(x[1] - convert_strip_length(-82.0u"nm"))^2)
    return a + b
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

plot(m_ref, gspec, 1)
plot!(m_opti_bil, gspec, 1)

##
plot!(rand(m_opti_bil), gspec, 1)
##

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

## PLOTTING
using LaTeXStrings
using Serialization
using Optim
opt_box = deserialize("scripts/reconst_1d_res_diff/box_diff.jls")
opt_bil = deserialize("scripts/reconst_1d_res_diff/bil_diff.jls")
opt_nn = deserialize("scripts/reconst_1d_results/nn.jls")

##

gspec = make_gridspec(
    (300, 1, 1),
    DTYPE.(convert_strip_length.((-500.0u"nm", 0.0u"nm"))),
    DTYPE.(convert_strip_length.(0u"nm")),
    DTYPE.(convert_strip_length.(0u"nm")))

p = Plots.plot(grid=:none, legend=:bottomright, size=(600*0.7, 400*0.7))

x_values = points(gspec.dims[1].e) ./ 1e-7
Plots.plot!(x_values, density_eee(m_ref, gspec)[:, 1, 1], color=:black, xflip=true, linestyle=:solid, xlims=(-305, 0), label="reference")
# Plots.vline!(range(-300, 0, length=10), color=:gray, alpha=0.3, label=:none)
Plots.xlabel!(L"x\quad \textrm{in\, [nm]}")
Plots.ylabel!(L"\rho\quad \textrm{in\, [g \, \, cm^{-3}]}")
for (i, linest) in zip([1, 2, 6, 10], [:dot, :dash, :dashdot, :solid])
    from_param_vec!(m_opti_box, opt_box[1].trace[i].metadata["x"])
    from_param_vec!(m_opti_bil, opt_bil[1].trace[i].metadata["x"])
    from_param_vec!(m_opti_nn, opt_nn[1].trace[i].metadata["x"])

    Plots.plot!(x_values, density_eee(m_opti_box, gspec)[:, 1, 1], color=1, xflip=true, linestyle=linest, label=string("iteration ", i-1))
    #Plots.plot!(x_values, density_eee(m_opti_bil, gspec)[:, 1, 1], color=2, xflip=true, linestyle=linest, label=string("iteration ", i-1))
    # Plots.plot!(x_values, density_eee(m_opti_nn, gspec)[:, 1, 1], color=3, xflip=true, linestyle=linest, label=string("iteration ", i-1))
end

# Plots.savefig("scripts/master_thesis_figures/1d_reconstruction/box_diff.pdf")
p
##

Plots.plot([res.value for res in opt_box[1].trace] ./ opt_box[1].trace[1].value, yaxis=:log, color=1, label="Piecewise-Constant", xlims=(0, 200), ylims=(1e-8, 100), size=(600*0.7, 400*0.7))
Plots.plot!([res.value for res in opt_bil[1].trace]./ opt_bil[1].trace[1].value, color=2, label="Bilinear")
Plots.plot!([res.value for res in opt_nn[1].trace] ./ opt_nn[1].trace[1].value, color=3, label="Neural Network")


Plots.plot!(opt_box[3] ./ opt_box[3][1], color=1, linestyle=:dash, label=:none)
Plots.plot!(opt_bil[3] ./ opt_bil[3][1], color=2, linestyle=:dash, label=:none)
Plots.plot!(opt_nn[3] ./ opt_nn[3][1], color=3, linestyle=:dash, label=:none)

Plots.plot!(opt_box[2] ./ opt_box[2][1], color=1, linestyle=:dot, label=:none)
Plots.plot!(opt_bil[2] ./ opt_bil[2][1], color=2, linestyle=:dot, label=:none)
Plots.plot!(opt_nn[2] ./ opt_nn[2][1], color=3, linestyle=:dot, label=:none)

Plots.plot!([], [], linestyle=:solid, color=:gray, label=L"||k(p) - k^{\textrm{exp}}||^2")
Plots.plot!([], [], linestyle=:dash, color=:gray, label=L"||\rho(x) - \rho^{\textrm{true}}||^2")

Plots.ylabel!(L"\textrm{error}")
Plots.xlabel!(L"\textrm{iteration}")
#Plots.savefig("scripts/master_thesis_figures/1d_reconstruction/losses.pdf")
##
# Plots.plot(opt_box[2], yaxis=:log, color=1, linestyle=:dash)
# Plots.plot!(opt_bil[2], yaxis=:log, color=2, linestyle=:dash)
# Plots.plot!(opt_nn[2], yaxis=:log, color=3, linestyle=:dash)
# ##