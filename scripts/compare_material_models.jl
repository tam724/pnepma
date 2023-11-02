include("../pn_solver/pn_parametrizations/pn_material.jl")
##
using Plots
using NeXLCore
using LaTeXStrings
include("../pn_solver/units.jl")

##

p = fill(BilinearPNParametrization{true, 2}, (0., 1.), (0., 0.), (0., 0.), 2, 1, 1, 0.5)
p.vals[1] = [1.0, 0.0]
p.vals[2] = [0.0, 1.0]

elms = [n"Pb", n"Si"]

m_mass_fraction = MassFractionPNMaterial(elms, p)
m_volume_fractions = VolumeFractionPNMaterial(elms, p)
m_linear_density = LinearDensityPNMaterial(elms, p)

X = 0:0.01:1

ρ_mass_fraction = [component_densities(m_mass_fraction, [x, 0., 0.]) for x in X]
ρ_full_mass_fraction = [density(m_mass_fraction, [x, 0., 0.]) for x in X]

ρ_volume_fraction = [component_densities(m_volume_fractions, [x, 0., 0.]) for x in X]
ρ_full_volume_fraction = [density(m_volume_fractions, [x, 0., 0.]) for x in X]

ρ_linear = [component_densities(m_linear_density, [x, 0., 0.]) for x in X]
ρ_full_linear = [density(m_linear_density, [x, 0., 0.]) for x in X]


##
Plots.plot([0,0], [0,0], label=L"\rho_{\textrm{tot}}", color=:green, size=(600*0.6, 400*0.6))
Plots.plot!([0,0], [0,0], label=L"\rho_{Pb}", color=:red)
Plots.plot!([0,0], [0,0], label=L"\rho_{Si}", color=:blue)

Plots.plot!([0,0], [0,0], label=L"\omega_i", color=:gray, linestyle=:dot)
Plots.plot!([0,0], [0,0], label=L"\phi_i", color=:gray, linestyle=:dash)
Plots.plot!([0,0], [0,0], label=L"\gamma_i", color=:gray, linestyle=:dashdot)

Plots.plot!(X, getindex.(ρ_mass_fraction, 1), label="", color=:red, linestyle=:dot)
Plots.plot!(X, getindex.(ρ_mass_fraction, 2), label="", color=:blue, linestyle=:dot)
Plots.plot!(X, ρ_full_mass_fraction, label="", color=:green, linestyle=:dot)

Plots.plot!(X, getindex.(ρ_volume_fraction, 1), label="", color=:red, linestyle=:dash)
Plots.plot!(X, getindex.(ρ_volume_fraction, 2), label="", color=:blue, linestyle=:dash)
Plots.plot!(X, ρ_full_volume_fraction, label="", color=:green, linestyle=:dash)

Plots.plot!(X, getindex.(ρ_linear, 1), label="", color=:red, linestyle=:dashdot)
Plots.plot!(X, getindex.(ρ_linear, 2), label="", color=:blue, linestyle=:dashdot)
Plots.plot!(X, ρ_full_linear, label="", color=:green, linestyle=:dashdot)

Plots.xlabel!(L"\omega_i / \phi_i / \gamma_i")
Plots.ylabel!(L"\rho_{\textrm{tot}} / \rho_i\quad \textrm{in\ [g\ cm^{-3}]}")

Plots.savefig("scripts/master_thesis_figures/density_parametrization_comparison.pdf")

## inversion

p = ConstPNParametrization{2}(fill(0.33, 2))

N = 200
elms = [n"Pb", n"Si", n"Cu"]

m_linear = LinearDensityPNMaterialSC(elms, p)
m_mass_fraction = MassFractionPNMaterialSC(elms, p)
m_volume_fraction = VolumeFractionPNMaterialSC(elms, p)

ρ_exp_linear = component_densities(m_linear, [])
ρ_exp_mass_fraction = component_densities(m_mass_fraction, [])
ρ_exp_volume_fraction = component_densities(m_volume_fraction, [])

LOSS_linear = fill(NaN, (N, N))
LOSS_mass_fraction = fill(NaN, (N, N))
LOSS_volume_fraction = fill(NaN, (N, N))

l = 0.
r = 1.
X = range(l, r, length=N)
Y = range(l, r, length=N)

loss(ρ1, ρ2) = (sum((ρ1 .- ρ2).^2))

for (i, x) in enumerate(X)
    for (j, y) in enumerate(Y)
        if i > N - j + 1 continue end 
        p.p[1] = x
        p.p[2] = y
        LOSS_linear[i, j] = loss(ρ_exp_linear, component_densities(m_linear, []))
        LOSS_mass_fraction[i, j] = loss(ρ_exp_mass_fraction, component_densities(m_mass_fraction, []))
        LOSS_volume_fraction[i, j] = loss(ρ_exp_volume_fraction, component_densities(m_volume_fraction, []))
    end
end

clims = (minimum(minimum.((filter(!isnan, LOSS_linear), filter(!isnan, LOSS_mass_fraction), filter(!isnan, LOSS_volume_fraction)))), maximum(maximum.((filter(!isnan, LOSS_linear), filter(!isnan, LOSS_mass_fraction), filter(!isnan, LOSS_volume_fraction)))))

clims = (0, 10)
p_lin = Plots.plot(X, Y, LOSS_linear, st=:contourf, legend=nothing, aspect_ratio=:equal, clims=clims)
Plots.xlabel!(L"p_{Pb}")
Plots.ylabel!(L"p_{Si}")
Plots.xlims!((l, r))
Plots.ylims!((l, r))
Plots.title!(L"Linear\, Density")
p_mass = Plots.plot(X, Y, LOSS_mass_fraction, st=:contourf, legend=nothing, aspect_ratio=:equal, clims=clims)
Plots.xlabel!(L"\omega_{Pb}")
Plots.ylabel!(L"\omega_{Si}")
Plots.xlims!((l, r))
Plots.ylims!((l, r))
Plots.title!(L"Mass\, Fraction")
p_vol = Plots.plot(X, Y, LOSS_volume_fraction, st=:contourf, legend=nothing, aspect_ratio=:equal, clims=clims)
Plots.xlabel!(L"\varphi_{Pb}")
Plots.ylabel!(L"\varphi_{Si}")
Plots.xlims!((l, r))
Plots.ylims!((l, r))
Plots.title!(L"Volume\, Fraction")
Plots.plot(p_mass, p_vol, p_lin, layout=(1, 3), size=(800, 300), margin=5Plots.mm)

# Plots.savefig("scripts/master_thesis_figures/density_parametrization_comparison_inverse.pdf")
