## STORES / CONVERTS TO CALCULATION UNITS


default_mass = u"g"
default_length = u"cm"
default_energy = u"eV"
default_volume = default_length^3

function convert_strip_unit(unit, value)
    return ustrip(uconvert(unit, value))
end

convert_strip_mass(value) = convert_strip_unit(default_mass, value)
convert_strip_length(value) = convert_strip_unit(default_length, value)
convert_strip_length_1(value) = convert_strip_unit(default_length^(-1), value)
convert_strip_energy(value) = convert_strip_unit(default_energy, value)
convert_strip_volume(value) = convert_strip_unit(default_volume, value)
convert_strip_density(value) = convert_strip_unit(default_mass/default_volume, value)
