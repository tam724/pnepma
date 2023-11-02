include("../pn_solver/pn_solver.jl")

using NeXLMatrixCorrection
import NeXLMatrixCorrection: XPP, CitZAF, XPhi, Riveros1993

##
using LaTeXStrings

##

function XPP(x_ray, E)
    elm = elements[x_ray.z]
    mat = Material("", Dict(elm => 1.0))
    return NeXLMatrixCorrection.XPP(mat, inner(x_ray), E)
end

function CitZAF(x_ray, E)
    elm = elements[x_ray.z]
    mat = Material("", Dict(elm => 1.0))
    return NeXLMatrixCorrection.CitZAF(mat, inner(x_ray), energy(inner(x_ray)), E)
end

function XPhi(x_ray, E)
    elm = elements[x_ray.z]
    mat = Material("", Dict(elm => 1.0))
    return NeXLMatrixCorrection.XPhi(mat, inner(x_ray), E)
end

function Riveros1993(x_ray, E)
    elm = elements[x_ray.z]
    mat = Material("", Dict(elm => 1.0))
    return NeXLMatrixCorrection.Riveros1993(mat, inner(x_ray), energy(inner(x_ray)), E)
end

## PROBLEM SETUP
DTYPE = Float64

gspec = make_gridspec(
    (200, 1, 1), 
    DTYPE.(convert_strip_length.((-1000.0u"nm", 0.0u"nm"))),
    DTYPE.(convert_strip_length.(0u"nm")),
    DTYPE.(convert_strip_length.(0u"nm")))

##

struct PN end 

function phi_rho_z(::Type{PN}, x_ray, beam_energy)
    E_init = beam_energy + 1u"keV"
    E_cut = edgeenergy(x_ray)*1u"eV" - 0.1u"keV"
    elms = [elements[x_ray.z]]
    x_rays = [x_ray]
    m = VolumeFractionPNMaterial(elms, ConstPNParametrization{1}([1.0]))
    problem = ForwardPNProblem{21, Float64}(
        gspec,
        elms,
        #beam_energy
        Normal(
            convert_strip_energy(beam_energy),
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
        2000);
    x_gen = calc_x_ray_generation_field(problem, m, x_rays)
    return getindex.(x_gen, 1)[:, 1, 1]
end


function phi_rho_z(::Type{T}, x_ray, beam_energy) where T
    method = T(x_ray, convert_strip_unit(u"eV", beam_energy))
    x = points(gspec.dims[1].e) .* default_length
    ρx = convert_strip_unit.(Ref(u"g"/u"cm"^2), x .* elements[x_ray.z].density)
    return ϕ.(Ref(method), -ρx)
end

##

x_rays = [characteristic(n"Si", ktransitions)...,
characteristic(n"Si", ltransitions)...,
characteristic(n"Cu", ktransitions)...,
characteristic(n"Cu", ltransitions)...,
characteristic(n"Ga", ktransitions)...,
characteristic(n"Ga", ltransitions)...,
characteristic(n"Fe", ktransitions)...,
characteristic(n"Fe", ltransitions)...]

##

for x_ray in x_rays
    for beam_energy in [5., 10., 15., 20., 25.] .* 1u"keV"
        if edgeenergy(x_ray) * 1u"eV" > beam_energy
            @show x_ray
            @show edgeenergy(x_ray)
            @show beam_energy
            continue
        end

        elm_symbol = element(x_ray.z).symbol
        inner_symbol = inner(x_ray).subshell
        outer_symbol = outer(x_ray).subshell
        beam_energy_keV = convert_strip_unit(u"keV", beam_energy)
        ##

        phirhoz_PN_Cu = phi_rho_z(PN, x_ray, beam_energy)
        phirhoz_XPP_Cu = phi_rho_z(XPP, x_ray, beam_energy)
        phirhoz_CitZAF_Cu = phi_rho_z(CitZAF, x_ray, beam_energy)
        phirhoz_XPhi_Cu = phi_rho_z(XPhi, x_ray, beam_energy)
        phirhoz_Riveros_Cu = phi_rho_z(Riveros1993, x_ray, beam_energy)

        ##
        x = points(gspec.dims[1].e) .* default_length
        Plots.plot(convert_strip_unit.(Ref(u"nm"), x), normalize(phirhoz_PN_Cu) , xflip=true, label="PN")
        Plots.plot!(convert_strip_unit.(Ref(u"nm"), x), normalize(phirhoz_XPP_Cu), xflip=true, label="XPP")
        Plots.plot!(convert_strip_unit.(Ref(u"nm"), x), normalize(phirhoz_CitZAF_Cu), xflip=true, label="CitZAF")
        Plots.plot!(convert_strip_unit.(Ref(u"nm"), x), normalize(phirhoz_XPhi_Cu), xflip=true, label="XPhi")
        Plots.plot!(convert_strip_unit.(Ref(u"nm"), x), normalize(phirhoz_Riveros_Cu), xflip=true, label="Riveros1993")
        Plots.title!(L"\textrm{%$elm_symbol _{%$inner_symbol \to %$outer_symbol}} \quad %$beam_energy_keV \textrm{keV}")
        Plots.ylabel!(L"\phi(x)\quad \textrm{(normalized)}")
        Plots.xlabel!(L"x\quad \textrm{in\, [nm]}")

        @show "scripts/master_thesis_figures/phi_rho_z_comparison/$elm_symbol$inner_symbol$(outer_symbol)E$beam_energy_keV.pdf"
        Plots.savefig("scripts/master_thesis_figures/phi_rho_z_comparison/$elm_symbol$inner_symbol$(outer_symbol)E$beam_energy_keV.pdf")
    end
end