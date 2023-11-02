include("../pn_solver/pn_solver.jl")
using LaTeXStrings

## PROBLEM SETUP
DTYPE = Float64

gspec = make_gridspec(
    (100, 1, 1), 
    DTYPE.(convert_strip_length.((-500.0u"nm", 0.0u"nm"))),
    DTYPE.(convert_strip_length.(0u"nm")),
    DTYPE.(convert_strip_length.(0u"nm")))

elms = @SVector [n"Cu", n"Ni"]
x_rays = @SVector[n"Cu K-L2", n"Cu L2-M3", n"Ni K-L3", n"Ni L2-M3"]

# m_names = ["Boxed", "Bilinear", "Neural Network"]
m_names = ["Boxed", "Bilinear", "Boxed 40", "Bilinear 40", "BoxedB", "BilinearB", "Boxed 40B", "Bilinear 40B"]

timings = []
errors = []
init() = [0.5]
for (m_i, m) in enumerate([
    MassFractionPNMaterialSC(elms,
        BoxedPNParametrization{false, 1}(
            [init() for _ in 1:20 for _ in 1:1 for _ in 1:1],
            [0.5],
            range(convert_strip_length(-500.0u"nm"), convert_strip_length(0.0u"nm"), length=21),
            range(convert_strip_length(-100.0u"nm"), convert_strip_length(100.0u"nm"), length=2),
            range(convert_strip_length(-100.0u"nm"), convert_strip_length(100.0u"nm"), length=2))
        ),
    MassFractionPNMaterialSC(elms,
        BilinearPNParametrization{false, 1}(
            [init() for _ in 1:20 for _ in 1:1 for _ in 1:1],
            [0.5],
            range(convert_strip_length(-500.0u"nm"), convert_strip_length(0.0u"nm"), length=20),
            range(convert_strip_length(0.0u"nm"), convert_strip_length(0.0u"nm"), length=1),
            range(convert_strip_length(0.0u"nm"), convert_strip_length(0.0u"nm"), length=1))
        ),
    MassFractionPNMaterialSC(elms,
        BoxedPNParametrization{false, 1}(
            [init() for _ in 1:40 for _ in 1:1 for _ in 1:1],
            [0.5],
            range(convert_strip_length(-500.0u"nm"), convert_strip_length(0.0u"nm"), length=41),
            range(convert_strip_length(-100.0u"nm"), convert_strip_length(100.0u"nm"), length=2),
            range(convert_strip_length(-100.0u"nm"), convert_strip_length(100.0u"nm"), length=2))
        ),
    MassFractionPNMaterialSC(elms,
        BilinearPNParametrization{false, 1}(
            [init() for _ in 1:40 for _ in 1:1 for _ in 1:1],
            [0.5],
            range(convert_strip_length(-500.0u"nm"), convert_strip_length(0.0u"nm"), length=40),
            range(convert_strip_length(0.0u"nm"), convert_strip_length(0.0u"nm"), length=1),
            range(convert_strip_length(0.0u"nm"), convert_strip_length(0.0u"nm"), length=1))
        ),
    MassFractionPNMaterialSC(elms,
        BoxedPNParametrization{false, 1}(
            [init() for _ in 1:20 for _ in 1:1 for _ in 1:1],
            [0.5],
            range(convert_strip_length(-500.0u"nm"), convert_strip_length(0.0u"nm"), length=21),
            range(convert_strip_length(-100.0u"nm"), convert_strip_length(100.0u"nm"), length=2),
            range(convert_strip_length(-100.0u"nm"), convert_strip_length(100.0u"nm"), length=2))
        ),
    MassFractionPNMaterialSC(elms,
        BilinearPNParametrization{false, 1}(
            [init() for _ in 1:20 for _ in 1:1 for _ in 1:1],
            [0.5],
            range(convert_strip_length(-500.0u"nm"), convert_strip_length(0.0u"nm"), length=20),
            range(convert_strip_length(0.0u"nm"), convert_strip_length(0.0u"nm"), length=1),
            range(convert_strip_length(0.0u"nm"), convert_strip_length(0.0u"nm"), length=1))
        ),
    MassFractionPNMaterialSC(elms,
        BoxedPNParametrization{false, 1}(
            [init() for _ in 1:40 for _ in 1:1 for _ in 1:1],
            [0.5],
            range(convert_strip_length(-500.0u"nm"), convert_strip_length(0.0u"nm"), length=41),
            range(convert_strip_length(-100.0u"nm"), convert_strip_length(100.0u"nm"), length=2),
            range(convert_strip_length(-100.0u"nm"), convert_strip_length(100.0u"nm"), length=2))
        ),
    MassFractionPNMaterialSC(elms,
        BilinearPNParametrization{false, 1}(
            [init() for _ in 1:40 for _ in 1:1 for _ in 1:1],
            [0.5],
            range(convert_strip_length(-500.0u"nm"), convert_strip_length(0.0u"nm"), length=40),
            range(convert_strip_length(0.0u"nm"), convert_strip_length(0.0u"nm"), length=1),
            range(convert_strip_length(0.0u"nm"), convert_strip_length(0.0u"nm"), length=1))
        ),
    # VolumeFractionPNMaterialSC(elms,
    #     NNPNParametrization{1, 1}(
    #         Chain(
    #             Normalization(convert_strip_length.((-500.0u"nm", 0.0u"nm"))),
    #             Dense(1, 10, Tanh()),
    #             Dense(10, 1, Sigmoid())
    #             )
    #     )
    # ),
    ])
        
    detector_position = convert_strip_length.([10.0u"nm", 0.0u"nm", 0.0u"nm"])

    E_init = 16.0u"keV"
    E_cut = minimum(edgeenergy.(x_rays))*1u"eV" - 0.1u"keV"
    problem = ForwardPNProblem{9, Float64}(
        gspec,
        elms,
        #beam_energy
        Normal(
            convert_strip_energy(15.0u"keV"),
            convert_strip_energy(0.1u"keV")),
        #beam_position
        MultivariateNormal(
            convert_strip_length.([0.0u"nm", 0.0u"nm", 0.0u"nm"]),
            convert_strip_length(30.0u"nm")),
        #beam_direction
        VonMisesFisher(
            [-1., 0., 0.], 50.
        ),
        #E_initial
        DTYPE(convert_strip_energy(E_init)),
        #E_cutoff
        DTYPE(convert_strip_energy(E_cut)),
        #N_steps
        700);


    function Zygote_jacobian(f, m)
        _, back = Zygote._pullback(f, m)
        return [back([1.0, 0.0, 0.0, 0.0]), back([0.0, 1.0, 0.0, 0.0]), back([0.0, 0.0, 1.0, 0.0]), back([0.0, 0.0, 0.0, 1.0])]
    end

    ##

    GC.gc()
    zygote_jac = Zygote_jacobian(m_ -> calc_intensities(problem, m_, x_rays, detector_position), m)
    GC.gc()
    temp = @elapsed Zygote_jacobian(m_ -> calc_intensities(problem, m_, x_rays, detector_position), m)
    @show temp
    timing_zygote = @elapsed Zygote_jacobian(m_ -> calc_intensities(problem, m_, x_rays, detector_position), m)
    GC.gc()
    finite_jac = jacobian_finite_differences(m_ -> calc_intensities(problem, m_, x_rays, detector_position), m)
    GC.gc()
    temp = @elapsed jacobian_finite_differences(m_ -> calc_intensities(problem, m_, x_rays, detector_position), m)
    @show temp
    timing_fin_diff = @elapsed jacobian_finite_differences(m_ -> calc_intensities(problem, m_, x_rays, detector_position), m)

    push!(timings, (timing_zygote, timing_fin_diff))

    ##
    p = Plots.plot()
    push!(errors, [])
    for (r, x_ray) in enumerate(x_rays)
        m_zygote = zero(m)
        from_named_tuple!(m_zygote, zygote_jac[r][2])
        push!(errors[m_i], abs.((param_vec(m_zygote) - finite_jac[r, :])./param_vec(m_zygote)))

        from_param_vec!(m_zygote, normalize(param_vec(m_zygote)))
        m_finite = from_param_vec(m, normalize(finite_jac[r, :]))

        p_eval_x = zeros(100)
        p_eval_zygote = [zeros(1) for _ in 1:100]
        p_eval_finite = [zeros(1) for _ in 1:100]
        eee = EvenOddClassification{Even, Even, Even}
        for (i, x) in enumerate(points(eee, gspec))
            parametrization!(p_eval_zygote[i], m_zygote.p, collect(x))
            parametrization!(p_eval_finite[i], m_finite.p, collect(x))
            p_eval_x[i] = x[1]
        end
        # @show p_eval_zygote
        
        Plots.plot!(p_eval_x / 1.e-7, normalize(getindex.(p_eval_zygote, 1)), color=r, xflip=true, label=string(x_ray))
        Plots.plot!(p_eval_x / 1.e-7, normalize(getindex.(p_eval_finite, 1)), color=:black, linestyle=:dash, xflip=true, label=:none)
    end
    Plots.xlabel!(L"x\quad \textrm{in\, [nm]}")
    Plots.ylabel!(L"P(x; \frac{\partial I}{\partial p}) \quad \textrm{(normalized)}")
    Plots.plot!([], [], color=:black, linestyle=:dash, label="finite differences")
    Plots.savefig(string("scripts/master_thesis_figures/sensitivity_1d/", m_names[m_i], ".pdf"))
end

##
mats = [
    VolumeFractionPNMaterialSC(elms,
        BoxedPNParametrization{false, 1}(
            [rand(1) for _ in 1:20 for _ in 1:1 for _ in 1:1],
            [0.5],
            range(convert_strip_length(-500.0u"nm"), convert_strip_length(0.0u"nm"), length=21),
            range(convert_strip_length(-100.0u"nm"), convert_strip_length(100.0u"nm"), length=2),
            range(convert_strip_length(-100.0u"nm"), convert_strip_length(100.0u"nm"), length=2))
        ),
    LinearDensityPNMaterialSC(elms,
        BilinearPNParametrization{false, 1}(
            [rand(1) for _ in 1:20 for _ in 1:1 for _ in 1:1],
            [0.5],
            range(convert_strip_length(-500.0u"nm"), convert_strip_length(0.0u"nm"), length=20),
            range(convert_strip_length(0.0u"nm"), convert_strip_length(0.0u"nm"), length=1),
            range(convert_strip_length(0.0u"nm"), convert_strip_length(0.0u"nm"), length=1))
        ),
    VolumeFractionPNMaterialSC(elms,
        NNPNParametrization{1, 1}(
            Chain(
                Normalization(convert_strip_length.((-500.0u"nm", 0.0u"nm"))),
                Dense(1, 10, Tanh()),
                Dense(10, 1, Sigmoid())
                )
        )
    )]