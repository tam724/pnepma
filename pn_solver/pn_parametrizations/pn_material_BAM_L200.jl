
## y is the coordinate the BAM L200 material is changing

struct BAML200PNMaterial{E} <: PNMaterial{3, Float64}
    elms::E
end

function BAML200PNMaterial()
    return BAML200PNMaterial([n"Al", n"Ga", n"As"])
end

# absolute thicknesses of the layers:
thicknesses = [
    repeat([80, 67], 12)...,
    691, # W1
    691, # W2
    293, # W3
    294, # W4
    293,
    372, 
    19.5, # W5
    372,
    195, # W6 # P2
    195, # W7 # P2
    195, 
    384,
    273*0.5, # P3
    273*0.5, # P3
    273*0.5,
    330,
    193*0.5, # P4
    193*0.5, # P4
    193*0.5,
    282,
    136*0.5, # P5
    136*0.5, # P5
    136*0.5,
    245,
    97*0.5, # P6
    97*0.5, # P6
    97*0.5
]*u"nm"

thicknesses = convert_strip_length.(thicknesses)

interfaces = [cumsum(thicknesses)...]
# material = repeat([:AlGaAs, :GaAs], length(interfaces) ÷ 2)

function get_material(::BAML200PNMaterial, x)
    if x < 0.
        return Val(:GaAs)
    elseif x > interfaces[end]
        return Val(:GaAs)
    else
        for (i, I) in enumerate(interfaces)
            if x < I
                return mod(i, 2) == 1 ? Val(:AlGaAs) : Val(:GaAs)
            end
        end
    end
end

function get_component_densities(::Val{:GaAs})
    ρ_Ga = convert_strip_density(n"Ga".density)
    ρ_As = convert_strip_density(n"As".density)
    ω_Ga = 0.4820
    ω_As = 0.5180
    #temp = 1. / (ω_Ga / ρ_Ga + ω_As / ρ_As)
    temp = ω_Ga * ρ_Ga + ω_As * ρ_As
    return @SVector [0., ω_Ga*temp, ω_As*temp]
end

function get_component_densities(::Val{:AlGaAs})
    ρ_Al = convert_strip_density(n"Al".density)
    ρ_Ga = convert_strip_density(n"Ga".density)
    ρ_As = convert_strip_density(n"As".density)
    ω_Al = 0.1646
    ω_Ga = 0.1823
    ω_As = 0.6531
    #temp = 1. / (ω_Al / ρ_Al + ω_Ga / ρ_Ga + ω_As / ρ_As)
    temp = ω_Al * ρ_Al + ω_Ga * ρ_Ga + ω_As * ρ_As
    return @SVector [ω_Al*temp, ω_Ga*temp, ω_As*temp]
end

function component_densities!(ρ::AbstractVector, m::BAML200PNMaterial, x::AbstractVector)
    mat = get_material(m, x[2])
    ρ .= get_component_densities(mat)
end
