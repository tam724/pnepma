include("../pn_solver/pn_solver.jl")

##
using LaTeXStrings
using Serialization

## PROBLEM SETUP
DTYPE = Float64
elms = @SVector [n"Al", n"Cu"]
x_rays = @SVector [n"Al K-L2", n"Cu K-L2"]

m = VolumeFractionPNMaterialSC(elms,
    fill(BoxedPNParametrization{true, 1}, 
        convert_strip_length.([-1000.0u"nm", -700.0u"nm", -400.0u"nm", -200.0u"nm", -50.0u"nm", 0.0u"nm"]),
        convert_strip_length.([-1.0u"nm", 1.0u"nm"]),
        convert_strip_length.([-1.0u"nm", 1.0u"nm"]),
        1.0)
    )
m.p.vals[1, 1, 1] .= 0.0
m.p.vals[2, 1, 1] .= 1.0
m.p.vals[3, 1, 1] .= 0.0
m.p.vals[4, 1, 1] .= 1.0
m.p.vals[5, 1, 1] .= 0.0

gspec = make_gridspec(
    (200, 1, 1), 
    DTYPE.(convert_strip_length.((-1000.0u"nm", 0.0u"nm"))),
    DTYPE.(convert_strip_length.(0u"nm")),
    DTYPE.(convert_strip_length.(0u"nm")))

###

gspec

detector_position = convert_strip_length.([100.0u"nm", 0.0u"nm", 0.0u"nm"])

problem_15_9 = ForwardPNProblem{9, Float64}(
        gspec,
        m.elms,
        #beam_energy
        Normal(
            convert_strip_energy(15.0u"keV"),
            convert_strip_energy(0.2u"keV")),
        #beam_position
        MultivariateNormal(
            convert_strip_length.([0.0u"nm", 0.0u"nm", 0.0u"nm"]),
            convert_strip_length(30.0u"nm")),
        #beam_direction
        VonMisesFisher(
            [-1., 0., 0.], 10.
        ),
        #E_initial
        DTYPE(convert_strip_energy(15.5u"keV")),
        #E_cutoff
        DTYPE(convert_strip_energy(1.0u"keV")),
        #N_steps
        2000);

problem_10_9 = ForwardPNProblem{9, Float64}(
        gspec,
        m.elms,
        #beam_energy
        Normal(
            convert_strip_energy(10.0u"keV"),
            convert_strip_energy(0.2u"keV")),
        #beam_position
        MultivariateNormal(
            convert_strip_length.([0.0u"nm", 0.0u"nm", 0.0u"nm"]),
            convert_strip_length(30.0u"nm")),
        #beam_direction
        VonMisesFisher(
            [-1., 0., 0.], 10.
        ),
        #E_initial
        DTYPE(convert_strip_energy(15.5u"keV")),
        #E_cutoff
        DTYPE(convert_strip_energy(1.0u"keV")),
        #N_steps
        2000);

problem_15_3 = ForwardPNProblem{3, Float64}(
    gspec,
    m.elms,
    #beam_energy
    Normal(
        convert_strip_energy(15.0u"keV"),
        convert_strip_energy(0.2u"keV")),
    #beam_position
    MultivariateNormal(
        convert_strip_length.([0.0u"nm", 0.0u"nm", 0.0u"nm"]),
        convert_strip_length(30.0u"nm")),
    #beam_direction
    VonMisesFisher(
        [-1., 0., 0.], 10.
    ),
    #E_initial
    DTYPE(convert_strip_energy(15.5u"keV")),
    #E_cutoff
    DTYPE(convert_strip_energy(1.0u"keV")),
    #N_steps
    2000);

problem_10_3 = ForwardPNProblem{3, Float64}(
    gspec,
    m.elms,
    #beam_energy
    Normal(
        convert_strip_energy(10.0u"keV"),
        convert_strip_energy(0.2u"keV")),
    #beam_position
    MultivariateNormal(
        convert_strip_length.([0.0u"nm", 0.0u"nm", 0.0u"nm"]),
        convert_strip_length(30.0u"nm")),
    #beam_direction
    VonMisesFisher(
        [-1., 0., 0.], 10.
    ),
    #E_initial
    DTYPE(convert_strip_energy(15.5u"keV")),
    #E_cutoff
    DTYPE(convert_strip_energy(1.0u"keV")),
    #N_steps
    2000);
##

ρ = density_eee(m, gspec)
p = Plots.plot(points(gspec.dims[1].e) ./ 1e-7, ρ[:, 1, 1], xflip=true)
Plots.xlabel!(L"x \quad \textrm{in\, [nm]}")

##
store_15_9 = compute_and_save(problem_15_9, m)
store_15_3 = compute_and_save(problem_15_3, m)
store_10_9 = compute_and_save(problem_10_9, m)
store_10_3 = compute_and_save(problem_10_3, m)

##
Plots.plot(points(gspec.dims[1].e) ./ 1e-7, store_15_9[100][2][:, 1, 1] * 1e230, xflip=true, color=1, linestyle=:dashdotdot, xlims=(-505, 5), label=:none, legend=:right)
Plots.plot!(points(gspec.dims[1].e) ./ 1e-7, store_15_3[100][2][:, 1, 1] * 1e230, xflip=true, color=2, linestyle=:dashdotdot, label=:none)

Plots.plot!(points(gspec.dims[1].e) ./ 1e-7, store_15_9[300][2][:, 1, 1] * 1e230, xflip=true, color=1, linestyle=:dashdot, label=:none)
Plots.plot!(points(gspec.dims[1].e) ./ 1e-7, store_15_3[300][2][:, 1, 1] * 1e230, xflip=true, color=2, linestyle=:dashdot, label=:none)

Plots.plot!(points(gspec.dims[1].e) ./ 1e-7, store_15_9[500][2][:, 1, 1] * 1e230, xflip=true, color=1, linestyle=:dash, label=:none)
Plots.plot!(points(gspec.dims[1].e) ./ 1e-7, store_15_3[500][2][:, 1, 1] * 1e230, xflip=true, color=2, linestyle=:dash, label=:none)

Plots.plot!(points(gspec.dims[1].e) ./ 1e-7, store_15_9[700][2][:, 1, 1] * 1e230, xflip=true, color=1, linestyle=:dot, label=:none)
Plots.plot!(points(gspec.dims[1].e) ./ 1e-7, store_15_3[700][2][:, 1, 1] * 1e230, xflip=true, color=2, linestyle=:dot, label=:none)

Plots.plot!([], [], color=1, label="N = 9")
Plots.plot!([], [], color=2, label="N = 3")
Plots.plot!([], [], color=:gray, linestyle=:dashdotdot, label="$(round(store_15_9[100][1]/1000, digits=1))keV")
Plots.plot!([], [], color=:gray, linestyle=:dashdot, label="$(round(store_15_9[300][1]/1000, digits=1))keV")
Plots.plot!([], [], color=:gray, linestyle=:dash, label="$(round(store_15_9[500][1]/1000, digits=1))keV")
Plots.plot!([], [], color=:gray, linestyle=:dot, label="$(round(store_15_9[700][1]/1000, digits=1))keV")
Plots.annotate!(-25, 4, L"Cu")
Plots.annotate!(-125, 4, L"Al")
Plots.annotate!(-300, 4, L"Cu")
Plots.annotate!(-450, 4, L"Al")

Plots.vline!([-50.0, -200.0, -400.0, -700.0], color=:black, linestyle=:dot, label=:none)
Plots.ylabel!(L"\psi_0^0(\epsilon, x)")
Plots.xlabel!(L"x\quad \textrm{in\, [nm]}")
Plots.title!(L"15 \, \textrm{keV}")

Plots.savefig("scripts/master_thesis_figures/1D_layered_psi_15.pdf")


##

Plots.plot(points(gspec.dims[1].e) ./ 1e-7, store_10_9[800][2][:, 1, 1] * 1e230, xflip=true, color=1, label=:none, linestyle=:dashdotdot, xlims=(-505, 5), legend=:right)
Plots.plot!(points(gspec.dims[1].e) ./ 1e-7, store_10_3[800][2][:, 1, 1] * 1e230, xflip=true, color=2, label=:none, linestyle=:dashdotdot)

Plots.plot!(points(gspec.dims[1].e) ./ 1e-7, store_10_9[1100][2][:, 1, 1] * 1e230, xflip=true, color=1, label=:none, linestyle=:dashdot)
Plots.plot!(points(gspec.dims[1].e) ./ 1e-7, store_10_3[1100][2][:, 1, 1] * 1e230, xflip=true, color=2, label=:none, linestyle=:dashdotdot)

Plots.plot!(points(gspec.dims[1].e) ./ 1e-7, store_10_9[1300][2][:, 1, 1] * 1e230, xflip=true, color=1, label=:none, linestyle=:dash)
Plots.plot!(points(gspec.dims[1].e) ./ 1e-7, store_10_3[1300][2][:, 1, 1] * 1e230, xflip=true, color=2, label=:none, linestyle=:dash)

Plots.plot!(points(gspec.dims[1].e) ./ 1e-7, store_10_9[1500][2][:, 1, 1] * 1e230, xflip=true, color=1, label=:none, linestyle=:dot)
Plots.plot!(points(gspec.dims[1].e) ./ 1e-7, store_10_3[1500][2][:, 1, 1] * 1e230, xflip=true, color=2, label=:none, linestyle=:dot)

Plots.plot!([], [], color=1, label="N = 9")
Plots.plot!([], [], color=2, label="N = 3")
Plots.plot!([], [], color=:gray, linestyle=:dashdotdot, label="$(round(store_10_9[800][1]/1000, digits=1))keV")
Plots.plot!([], [], color=:gray, linestyle=:dashdot, label="$(round(store_10_9[1100][1]/1000, digits=1))keV")
Plots.plot!([], [], color=:gray, linestyle=:dash, label="$(round(store_10_9[1300][1]/1000, digits=1))keV")
Plots.plot!([], [], color=:gray, linestyle=:dot, label="$(round(store_10_9[1500][1]/1000, digits=1))keV")
Plots.annotate!(-25, 4, L"Cu")
Plots.annotate!(-125, 4, L"Al")
Plots.annotate!(-300, 4, L"Cu")
Plots.annotate!(-450, 4, L"Al")

Plots.vline!([-50.0, -200.0, -400.0, -700.0], color=:black, linestyle=:dot, label=:none)
Plots.ylabel!(L"\psi_0^0(\epsilon, x)")
Plots.xlabel!(L"x\quad \textrm{in\, [nm]}")
Plots.title!(L"10 \, \textrm{keV}")

Plots.savefig("scripts/master_thesis_figures/1D_layered_psi_10.pdf")


## X ray generation

x_gen_15_9 = calc_x_ray_generation_field(problem_15_9, m, x_rays)
x_gen_15_3 = calc_x_ray_generation_field(problem_15_3, m, x_rays)
x_gen_10_9 = calc_x_ray_generation_field(problem_10_9, m, x_rays)
x_gen_10_3 = calc_x_ray_generation_field(problem_10_3, m, x_rays)

##
p = Plots.plot()
# pts = [0; cumsum(diff(points(gspec.dims[1].e) ./ 1e-7) ./ ρ[2:end, 1, 1])]
# pts .-= maximum(pts)

##
p = Plots.plot(points(gspec.dims[1].e) ./ 1e-7, getindex.(x_gen_15_9, 1)[:, 1, 1]*1e249, label="Al K-L2, 15keV", xflip=true, ylim=(-0.1, 7), color=1, legend=:right)
p = Plots.plot!(points(gspec.dims[1].e) ./ 1e-7, getindex.(x_gen_15_9, 2)[:, 1, 1]*1e249, label="Cu K-L2, 15keV", xflip=true, color=2)

p = Plots.plot!(points(gspec.dims[1].e) ./ 1e-7, getindex.(x_gen_15_3, 1)[:, 1, 1]*1e249, label=:none, xflip=true, linestyle=:dash, color=1)
p = Plots.plot!(points(gspec.dims[1].e) ./ 1e-7, getindex.(x_gen_15_3, 2)[:, 1, 1]*1e249, label=:none, xflip=true, linestyle=:dash, color=2)

p = Plots.plot!(points(gspec.dims[1].e) ./ 1e-7, getindex.(x_gen_10_9, 1)[:, 1, 1]*1e249, label="Al K-L2, 10keV", xflip=true, color=3)
p = Plots.plot!(points(gspec.dims[1].e) ./ 1e-7, getindex.(x_gen_10_9, 2)[:, 1, 1]*1e249, label="Cu K-L2, 10keV", xflip=true, color=4)

p = Plots.plot!(points(gspec.dims[1].e) ./ 1e-7, getindex.(x_gen_10_3, 1)[:, 1, 1]*1e249, label=:none, xflip=true, linestyle=:dash, color=3)
p = Plots.plot!(points(gspec.dims[1].e) ./ 1e-7, getindex.(x_gen_10_3, 2)[:, 1, 1]*1e249, label=:none, xflip=true, linestyle=:dash, color=4)


Plots.plot!([], [], color=:gray, linestyle=:solid, label="N = 9")
Plots.plot!([], [], color=:gray, linestyle=:dash, label="N = 3")
Plots.annotate!(-15, 6.5, L"Cu")
Plots.annotate!(-125, 6.5, L"Al")
Plots.annotate!(-300, 6.5, L"Cu")
Plots.annotate!(-550, 6.5, L"Al")
Plots.annotate!(-800, 6.5, L"Cu")

Plots.vline!([-50.0, -200.0, -400.0, -700.0], color=:black, linestyle=:dot, label=:none)
Plots.ylabel!(L"\phi(x)\quad \textrm{(normalized)}")
Plots.xlabel!(L"x\quad \textrm{in\, [nm]}")

Plots.savefig("scripts/master_thesis_figures/1D_layered_phi.pdf")
