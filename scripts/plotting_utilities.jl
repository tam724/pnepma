using Plots
using LaTeXStrings
using NeXLCore

function Lstring(x_ray::CharXRay)
    elm_symbol = element(x_ray.z).symbol
    inner_symbol = inner(x_ray).subshell
    outer_symbol = outer(x_ray).subshell
    return L"\textrm{%$elm_symbol _{%$inner_symbol \to %$outer_symbol}}"
end

function energy_LString(beam_energy_keV::Number)
    e = round(beam_energy_keV, digits=2)
    return L"%$e \, \textrm{keV}"
end