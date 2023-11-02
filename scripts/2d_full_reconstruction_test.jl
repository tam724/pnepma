include("../pn_solver/pn_solver.jl")

##
using Optim
using Serialization

##
include("plotting_utilities.jl")

## PROBLEM SETUP
DTYPE = Float64
elms = @SVector [n"Cu", n"Fe"]
x_rays = @SVector [n"Cu K-L2", n"Cu K-L3", n"Fe K-L2", n"Fe K-L3"]

rot = -0.6
a_ = 4
b_ = 1
m = VolumeFractionPNMaterialSC(elms,
    InclusionPNParametrization{1}(
        convert_strip_length.([-20.0u"nm", 30.0u"nm", 0.0u"nm"]),
        [a_*cos(rot) -b_*sin(rot) 0
        a_*sin(rot) b_*cos(rot) 0
        0 0 1],
        convert_strip_length(50.0u"nm"),
        [1.0],
        [0.0]
        )
    )

##

m_std_Cu = VolumeFractionPNMaterial(elms, ConstPNParametrization{2}([1.0, 0.0]))
m_std_Fe = VolumeFractionPNMaterial(elms, ConstPNParametrization{2}([0.0, 1.0]))

gspec = make_gridspec(
    (80, 80, 1), 
    DTYPE.(convert_strip_length.((-300.0u"nm", 0.0u"nm"))),
    DTYPE.(convert_strip_length.(((-150.)u"nm", (150.)u"nm"))),
    DTYPE.(convert_strip_length.(0u"nm")))

contourf(m, gspec)

##

detector_position = convert_strip_length.([100.0u"nm", 0.0u"nm", 0.0u"nm"])

N = 5
problems = [ForwardPNProblem{N, Float64}(
        gspec,
        m.elms,
        #beam_energy
        Normal(
            convert_strip_energy(12.0u"keV"),
            convert_strip_energy(0.2u"keV")),
        #beam_position
        MultivariateNormal(
            convert_strip_length.([0.0u"nm", b_pos_y, 0.0u"nm"]),
            convert_strip_length(20.0u"nm")),
        #beam_direction
        VonMisesFisher(
            [-1., 0., 0.], 10.
        ),
        #E_initial
        DTYPE(convert_strip_energy(12.5u"keV")),
        #E_cutoff
        DTYPE(convert_strip_energy(7.0u"keV")),
        #N_steps
        300) for b_pos_y in [-50.0u"nm", -25.0u"nm", 0.0u"nm", 25.0u"nm", 50.0u"nm"]]

problem_std = ForwardPNProblem{N, Float64}(
    gspec,
    m.elms,
    #beam_energy
    Normal(
        convert_strip_energy(12.0u"keV"),
        convert_strip_energy(0.2u"keV")),
    #beam_position
    MultivariateNormal(
        convert_strip_length.([0.0u"nm", 0.0u"nm", 0.0u"nm"]),
        convert_strip_length(20.0u"nm")),
    #beam_direction
    VonMisesFisher(
        [-1., 0., 0.], 10.
    ),
    #E_initial
    DTYPE(convert_strip_energy(12.5u"keV")),
    #E_cutoff
    DTYPE(convert_strip_energy(7.0u"keV")),
    #N_steps
    300);
##

x_ray_gen = calc_x_ray_generation_field(problem_std, m_std_Cu, x_rays)
Plots.heatmap(getindex.(x_ray_gen, 1)[:, :, 1])

##
std_ints_Cu = calc_intensities(problem_std, m_std_Cu, x_rays, detector_position)
std_ints_Fe = calc_intensities(problem_std, m_std_Fe, x_rays, detector_position)

##
std_ints = [std_ints_Cu[1], std_ints_Cu[2], std_ints_Fe[3], std_ints_Fe[4]]

##
function measure(m)
    vcat(
        calc_intensities(problems[1], m, x_rays, detector_position) ./ std_ints,
        calc_intensities(problems[2], m, x_rays, detector_position) ./ std_ints,
        calc_intensities(problems[3], m, x_rays, detector_position) ./ std_ints,
        calc_intensities(problems[4], m, x_rays, detector_position) ./ std_ints,
        calc_intensities(problems[5], m, x_rays, detector_position) ./ std_ints,
    )
end

##

function residuum(m, measurement)
    return vcat(
        calc_intensities(problems[1], m, x_rays, detector_position) ./ std_ints - measurement[1:4],
        calc_intensities(problems[2], m, x_rays, detector_position) ./ std_ints - measurement[5:8],
        calc_intensities(problems[3], m, x_rays, detector_position) ./ std_ints - measurement[9:12],
        calc_intensities(problems[4], m, x_rays, detector_position) ./ std_ints - measurement[13:16],
        calc_intensities(problems[5], m, x_rays, detector_position) ./ std_ints - measurement[17:20],
    )
end

using Zygote:@adjoint
@adjoint function residuum(m, measurement)
    res = vcat(
        calc_intensities(problems[1], m, x_rays, detector_position) ./ std_ints - measurement[1:4],
        calc_intensities(problems[2], m, x_rays, detector_position) ./ std_ints - measurement[5:8],
        calc_intensities(problems[3], m, x_rays, detector_position) ./ std_ints - measurement[9:12],
        calc_intensities(problems[4], m, x_rays, detector_position) ./ std_ints - measurement[13:16],
        calc_intensities(problems[5], m, x_rays, detector_position) ./ std_ints - measurement[17:20],
    )
    function residuum_pullback(res_)
        # this could be implemented more efficiently, (solve 1 adjoint pde per x-ray line) although then we would have to adapt the grids
        # checkpointing (we calculate each forward pass again)
        _, back = Zygote._pullback(calc_intensities, problems[1], m, x_rays, detector_position)
        m_= back(res_[1:4] ./ std_ints)[3]
        _, back = Zygote._pullback(calc_intensities, problems[2], m, x_rays, detector_position)
        m_= Zygote.accum(m_, back(res_[5:8] ./ std_ints)[3])
        _, back = Zygote._pullback(calc_intensities, problems[3], m, x_rays, detector_position)
        m_ = Zygote.accum(m_, back(res_[9:12] ./ std_ints)[3])
        _, back = Zygote._pullback(calc_intensities, problems[4], m, x_rays, detector_position)
        m_ = Zygote.accum(m_, back(res_[13:16] ./ std_ints)[3])
        _, back = Zygote._pullback(calc_intensities, problems[5], m, x_rays, detector_position)
        m_ = Zygote.accum(m_, back(res_[17:20] ./ std_ints)[3])
        return m_, nothing
    end
    return res, residuum_pullback
end

Zygote.pullback(loss, m_opti, k_exp)

_, back = Zygote._pullback(loss, m_opti, k_exp)
back(1.0)

function loss(m, measurement)
    res = residuum(m, measurement)
    return sum(res.^2) / length(res)
end



Zygote.gradient(loss, m_opti, k_exp)

##

function loss_fg!(F, G, x) 
    # from_param_vec!(m_opti, clamp.(x, 0., 1.))
    from_param_vec!(m_opti, x)
    @show m_opti
    #display(plot(m_opti, gspec, 1))
    if G !== nothing
        loss_, back = Zygote.pullback(m_ -> loss(m_, k_exp), m_opti)
        #temp = Zygote.withgradient(m_ -> loss(m_, k_exp), m_opti)
        #loss_ = temp.val
        (m_grad, ) = back(1.0)
        from_named_tuple!(m_opti_grad, m_grad)
        param_vec!(G, m_opti_grad)
        return loss_
    end
    if F !== nothing
        loss_ = loss(m_opti, k_exp)
        return loss_
    end
end

##
k_exp = measure(m)

## 
# m_opti = VolumeFractionPNMaterialSC(elms,
#     NNPNParametrization{1, 2}(Chain(
#         SingleDistance(convert_strip_length.([-20.0u"nm", 30.0u"nm"]), convert_strip_length(50u"nm")),
#         Dense(1, 1, Sigmoid())
#     ))
# )
m_opti = VolumeFractionPNMaterialSC(elms,
    NNPNParametrization{1, 2}(
        Chain(
            Normalization(
                convert_strip_length.((-300.0u"nm", 0.0u"nm")),
                convert_strip_length.(((-150.)u"nm", (150.)u"nm"))
            ),
            EllipseLayer(
                [1., 0.3],
                [0.5, 0.5],
                [0.]
            ),
            Dense(1, 1, Sigmoid())
        )
    )
)
m_opti.p.chain.layers[3].weight .= -0.5
m_opti_grad = zero(m_opti)

##

function callback(x)
    from_param_vec!(m_opti, x[end].metadata["x"])

    serialize("2d_ellipse_results.jls", x)
    #display(contourf(m_opti, gspec))
    #display(Plots.plot(Plots.heatmap(density_eee(m_opti, gspec_plot)[:, :, 1], clims=clims), Plots.heatmap(density_eee(m_measure, gspec_plot)[:, :, 1], clims=clims)))
    
    # L_test = loss_test(m_opti, k_exp_test)
    # L_L2 = l2_loss(m_opti, l2_exp)
    # push!(my_loss_test, L_test)
    # push!(my_l2_loss, L_L2)
    return false
end
p = param_vec(m_opti)
opt = optimize(Optim.only_fg!(loss_fg!), p, LBFGS(), Optim.Options(store_trace=true, show_trace=true, extended_trace=true, callback=callback, iterations=3000))
