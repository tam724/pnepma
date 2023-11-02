include("../pn_solver/pn_solver.jl")
include("../pn_solver/pn_parametrizations/pn_material_BAM_L200.jl")

using Serialization

##
include("plotting_utilities.jl")

## PROBLEM SETUP
DTYPE = Float64
elms = @SVector [n"Si", n"Cu"]
x_rays = [characteristic(n"Si", ktransitions)..., characteristic(n"Si", ltransitions)..., characteristic(n"Cu", ktransitions)..., characteristic(n"Cu", ltransitions)...]
x_rays = SVector{26, CharXRay}(x_rays)

m = VolumeFractionPNMaterialSC(elms,
    fill(BoxedPNParametrization{true, 1}, 
        convert_strip_length.([-1000.0u"nm", -100.0u"nm", 0.0u"nm"]),
        convert_strip_length.([-1000.0u"nm", 0.0u"nm", 1000.0u"nm"]),
        convert_strip_length.([-1000.0u"nm", 0.0u"nm", 1000.0u"nm"]),
        1.0)
    )
m.p.vals[1, 1, 1] .= 1.0
m.p.vals[1, 1, 2] .= 0.0
m.p.vals[1, 2, 1] .= 0.0
m.p.vals[1, 2, 2] .= 1.0

m.p.vals[2, 1, 1] .= 0.0
m.p.vals[2, 1, 2] .= 1.0
m.p.vals[2, 2, 1] .= 1.0
m.p.vals[2, 2, 2] .= 0.0

gspec = make_gridspec(
    (150, 150, 1), 
    DTYPE.(convert_strip_length.((-500.0u"nm", 0.0u"nm"))),
    DTYPE.(convert_strip_length.(((-300.)u"nm", (300.)u"nm"))),
    DTYPE.(convert_strip_length.(0u"nm")))

##

detector_position = convert_strip_length.([-10000.0u"nm", 4000.0u"nm", 0.0u"nm"])

N = 21
problem = ForwardPNProblem{N, Float64}(
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
        2500);

##

ρ = density_eee(m, gspec)
p = Plots.plot(points(gspec.dims[2].e) ./ 1e-7 , points(gspec.dims[1].e) ./ 1e-7, ρ[:, :, 1], st=:heatmap, aspect_ratio=:equal, xlims=(-310, 310), ylims=(-510, 10), clims=(0, 11))
Plots.ylabel!(L"x\quad \textrm{in\, [nm]}")
Plots.xlabel!(L"y\quad \textrm{in\, [nm]}")
Plots.annotate!((-150, -50, L"Cu"))
Plots.annotate!((-150, -300, L"Si", :white))
Plots.annotate!((150, -50, L"Si", :white))
Plots.annotate!((150, -300, L"Cu"))
path = mkpath(string("scripts/master_thesis_figures/psi_2d_sico"))
savefig(string(path, "/material.pdf"))
##

 # animated_plot(problem::ForwardPNProblem{N, T}, m, type=:contourf) where {N, T}
gspec = grid(problem)
u = zeros(MPNSolverVariable{N, DTYPE}, gspec)
du = zeros(MPNSolverVariable{N, DTYPE}, gspec)
ρ = component_densities(m, gspec)
ΔE = (E_initial(problem) - E_cutoff(problem))/N_steps(problem)
@show ΔE
global E = E_initial(problem)
@show E

for i in 1:N_steps(problem)
    global E = step_pn!(problem, ρ, u, du, E, ΔE)
    @show compute_dt_max(problem, 1.0), ΔE, E
    #plot([plot(u.eee[i, :, 1, 1], title=string(u.moments.eee[i])) for i in 1:9]..., layout=(3, 3))
    if i%50 == 0
        p = Plots.plot(points(gspec.dims[2].e) ./ 1e-7 , points(gspec.dims[1].e) ./ 1e-7, max.(getindex.(u.eee, 1)[:, :, 1], 0), st=:contourf, legend=:none, aspect_ratio=:equal, xlims=(-300, 300), ylims=(-500, 0))
        E_title = string(round(E/1000., digits=2), "keV")
        Plots.title!(energy_LString(E/1000.))
        Plots.ylabel!(L"x\quad \textrm{in\, [nm]}")
        Plots.xlabel!(L"y\quad \textrm{in\, [nm]}")
        path = mkpath(string("scripts/master_thesis_figures/psi_2d_sico/PN", N))
        savefig(string(path, "/", E_title, ".pdf"))
    end
end

## X ray generation
x_gen = calc_x_ray_generation_field(problem, m, x_rays)

for (i, x_ray) in enumerate(x_rays)
    p = Plots.plot(points(gspec.dims[2].e) ./ 1e-7 , points(gspec.dims[1].e) ./ 1e-7, getindex.(x_gen, i)[:, :, 1], st=:contourf, legend=:none, aspect_ratio=:equal, xlims=(-300, 300), ylims=(-500, 0))
    Plots.title!(Lstring(x_ray))
    Plots.ylabel!(L"x\quad \textrm{in\, [nm]}")
    Plots.xlabel!(L"y\quad \textrm{in\, [nm]}")
    path = mkpath(string("scripts/master_thesis_figures/phi_2d_sico/PN", N))
    Plots.savefig(string(path, "/", string(x_ray), ".pdf"))
end