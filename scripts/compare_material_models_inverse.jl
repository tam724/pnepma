include("../pn_solver/pn_parametrizations/pn_material.jl")
##
using Plots
using NeXLCore
using LaTeXStrings
include("../pn_solver/units.jl")

## inversion

p = ConstPNParametrization{2}(fill(1. / 3., 2))

N = 200
elms = [n"Pb", n"Si", n"Cu"]

##

m_linear = LinearDensityPNMaterialSC(elms, p)
m_mass_fraction = MassFractionPNMaterialSC(elms, p)
m_volume_fraction = VolumeFractionPNMaterialSC(elms, p)

##
ρ_exp_linear = component_densities(m_linear, [])
ρ_exp_mass_fraction = component_densities(m_mass_fraction, [])
ρ_exp_volume_fraction = component_densities(m_volume_fraction, [])

loss(ρ1, ρ2) = (sum((ρ1 .- ρ2).^2))

##

hess_linear = ForwardDiff_hessian(m -> loss(ρ_exp_linear, component_densities(m, [])), m_linear)
hess_mass_fraction = ForwardDiff_hessian(m -> loss(ρ_exp_mass_fraction, component_densities(m, [])), m_mass_fraction)
hess_volume_fraction = ForwardDiff_hessian(m -> loss(ρ_exp_volume_fraction, component_densities(m, [])), m_volume_fraction)

min_linear, max_linear = extrema(eigen(hess_linear).values)
min_mass, max_mass = extrema(eigen(hess_mass_fraction).values)
min_volume, max_volume = extrema(eigen(hess_volume_fraction).values)

@show max_linear/min_linear, max_mass/min_mass, max_volume/min_volume


##

LOSS_linear = fill(NaN, (N, N))
LOSS_mass_fraction = fill(NaN, (N, N))
LOSS_volume_fraction = fill(NaN, (N, N))

l = 0. # 1. / 3. - 0.03
r = 1. # / 3. + 0.03
X = range(l, r, length=N)
Y = range(l, r, length=N)

for (i, x) in enumerate(X)
    for (j, y) in enumerate(Y)
        if x + y > 1 continue end 
        p.p[1] = x
        p.p[2] = y
        LOSS_linear[i, j] = loss(ρ_exp_linear, component_densities(m_linear, []))
        LOSS_mass_fraction[i, j] = loss(ρ_exp_mass_fraction, component_densities(m_mass_fraction, []))
        LOSS_volume_fraction[i, j] = loss(ρ_exp_volume_fraction, component_densities(m_volume_fraction, []))
    end
end

clims = (minimum(minimum.((filter(!isnan, LOSS_linear), filter(!isnan, LOSS_mass_fraction), filter(!isnan, LOSS_volume_fraction)))), maximum(maximum.((filter(!isnan, LOSS_linear), filter(!isnan, LOSS_mass_fraction), filter(!isnan, LOSS_volume_fraction)))))

clims = (0., 10)
p_lin = Plots.plot(X, Y, LOSS_linear, st=:contourf, legend=nothing, aspect_ratio=:equal, clims=clims)
Plots.xlabel!(L"\gamma_{Pb}")
Plots.ylabel!(L"\gamma_{Si}")
Plots.xlims!((l, r))
Plots.ylims!((l, r))
Plots.title!(L"\textrm{Linear\, Density}")
p_mass = Plots.plot(X, Y, LOSS_mass_fraction, st=:contourf, legend=nothing, aspect_ratio=:equal, clims=clims)
Plots.xlabel!(L"\omega_{Pb}")
Plots.ylabel!(L"\omega_{Si}")
Plots.xlims!((l, r))
Plots.ylims!((l, r))
Plots.title!(L"\textrm{Mass\, Fraction}")
p_vol = Plots.plot(X, Y, LOSS_volume_fraction, st=:contourf, legend=nothing, aspect_ratio=:equal, clims=clims)
Plots.xlabel!(L"\varphi_{Pb}")
Plots.ylabel!(L"\varphi_{Si}")
Plots.xlims!((l, r))
Plots.ylims!((l, r))
Plots.title!(L"\textrm{Volume\, Fraction}")
Plots.plot(p_mass, p_vol, p_lin, layout=(1, 3), size=(800, 300), margin=5Plots.mm)

Plots.savefig("scripts/master_thesis_figures/density_parametrization_comparison_inverse.pdf")
