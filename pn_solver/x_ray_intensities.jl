

import NeXLCore: mac

function PNXRay(x_ray::CharXRay, els::AbstractVector{Element})
    mac_(e::Element) = mac(e, x_ray, FFASTDB)
    return PNXRay(x_ray, mac_.(els))
end

"""
emission cross secion.

Returns the emission cross section for the x_ray at electron energy (in eV)
"""

# function ion(electron_energy, x_ray)
#     z = element(x_ray).number
#     return @edit ionizationcrosssection(z, inner(x_ray).subshell.index, electron_energy, NeXLCore.Bote2009)
# end

# function flue(x_ray)
#     z = element(x_ray).number
#     return fluorescenceyield(z, inner(x_ray).subshell.index, outer(x_ray).subshell.index, NeXL)
# end

function collect!(x::AbstractArray{T}, y::Tuple{T, T, T}) where {T}
    x[1] = y[1]
    x[2] = y[2]
    x[3] = y[3]
end

function emissioncrosssection(electron_energy, x_ray)
    z = element(x_ray).number
    #return ionizationcrosssection(z, inner(x_ray).subshell.index, electron_energy, NeXLCore.Bote2009)
    return ionizationcrosssection(z, inner(x_ray).subshell.index, electron_energy, NeXLCore.Bote2009) # *
       #fluorescenceyield(z, inner(x_ray).subshell.index, outer(x_ray).subshell.index, NeXLCore.CullenEADL)
end

"""
ρ_i * N_A/A_i
"""

function compute_number_of_atoms(m::PNMaterial, gspec::GridSpec)
    eee = EvenOddClassification{Even, Even, Even}
    atomic_mass(el::Element) = convert_strip_mass(el.atomic_mass)
    AM = atomic_mass.(m.elms)

    number_of_atoms = [zeros(n_elements(m)) for _ in 1:size(eee, gspec, 1), _ in 1:size(eee, gspec, 2), _ in 1:size(eee, gspec, 3)]
    # for (noa, point) in zip(number_of_atoms, points(eee, gspec))
    for (i, point) in enumerate(points(eee, gspec))
        component_densities!(number_of_atoms[i], m, collect(point))
        number_of_atoms[i] ./= AM  
    end
    return number_of_atoms
end

function compute_number_of_atoms!(number_of_atoms, AM, m::PNMaterial, gspec::GridSpec)
    eee = EvenOddClassification{Even, Even, Even}
    # atomic_mass(el::Element) = convert_strip_mass(el.atomic_mass)
    # AM = convert_strip_mass.(atomic_mass.(m.elms))
    
    # for (noa, point) in zip(number_of_atoms, points(eee, gspec))
    x_ = @MVector [0., 0., 0.]
    @inbounds for (i, point) in enumerate(points(eee, gspec))
        collect!(x_, point)
        component_densities!(number_of_atoms[i], m, x_)
        number_of_atoms[i] ./= AM
    end
    return number_of_atoms
end

function compute_number_of_atoms!_pullback(m::PNMaterial, AM, gspec::GridSpec, m_, number_of_atoms_)
    eee = EvenOddClassification{Even, Even, Even}
    # atomic_mass(el::Element) = convert_strip_mass(el.atomic_mass)
    # AM = atomic_mass.(m.elms)  ### use as input

    x_ = @MVector [0., 0., 0.]
    @inbounds for (i, point) in enumerate(points(eee, gspec))
        collect!(x_, point)
        number_of_atoms_[i] ./= AM
        component_densities_pullback!(nothing, m, x_, number_of_atoms_[i], m_)
    end
end

function compute_number_of_atoms_pullback(m::PNMaterial, gspec::GridSpec, m_::PNMaterial, number_of_atoms_)
    eee = EvenOddClassification{Even, Even, Even}
    atomic_mass(el::Element) = convert_strip_mass(el.atomic_mass)
    AM = atomic_mass.(m.elms)  ### use as input

    x_ = @MVector [0., 0., 0.]
    for (i, point) in enumerate(points(eee, gspec))
        collect!(x_, point)
        number_of_atoms_[i] ./= AM
        component_densities_pullback!(nothing, m, x_, number_of_atoms_[i], m_)
    end
end

##

function compute_absorption_coefficient!(ac, dir, macs, origin, n_int_points, ρ, point, m::PNMaterial, x_rays, x_d, x_b)
    dir .= x_d .- origin
    λ = max((x_b[1] - origin[1])/ dir[1], 0.)
    dx = norm(dir)*λ/n_int_points

    for l in range(0., λ, length=n_int_points)
        point .= origin .+ l .* dir
        component_densities!(ρ, m, point)
        for r in 1:length(x_rays)
            ac[r] -= dot(macs[r], ρ)
        end
    end
    ac .= exp.(ac.*dx)
end

function compute_absorption_coefficients(m::PNMaterial, x_rays, gspec, x_d, x_b)
    eee = EvenOddClassification{Even, Even, Even}
    macs = [[mac(e, x_ray) for e in m.elms] for x_ray in x_rays] # NOTE: vec of vec for efficiency reasons
    acs = [zeros(length(x_rays)) for _ in 1:size(eee, gspec, 1), _ in 1:size(eee, gspec, 2), _ in 1:size(eee, gspec, 3)]
    dir = zero(x_d)
    n_int_points = 100
    point = @MVector [0.0, 0.0, 0.0]
    ρ = zeros(length(m.elms))

    for (acs_, origin) in zip(acs, points(eee, gspec))
        compute_absorption_coefficient!(acs_, dir, macs, origin, n_int_points, ρ, point, m, x_rays, x_d, x_b)
    end
    return acs
end


function compute_absorption_coefficient_pullback(m::PNMaterial, x_rays, gspec, x_d, x_b, acs, m_adj, acs_adj)
    macs = [mac(e, x_ray) for e in m.elms, x_ray in x_rays] # NOTE matrix for efficiency reasons
    dir = @MVector [0.0, 0.0, 0.0]
    n_int_points = 100
    point = @MVector [0.0, 0.0, 0.0]
    ρ_adj = zeros(n_elements(m))
    eee = EvenOddClassification{Even, Even, Even}

    # to follow the adjoint equations this should be implemented differently (integrate to the reflection)
    for (acs_, acs_adj_, origin) in zip(acs, acs_adj, points(eee, gspec))
        dir .= x_d .- origin
        λ = max((x_b[1] - origin[1])/ dir[1], 0.)
        dx = norm(dir)*λ/n_int_points

        acs_adj_ .= -acs_ .* dx .* acs_adj_
    
        for l in range(0., λ, length=n_int_points)
            point .= origin .+ l .* dir
            mul!(ρ_adj, macs, acs_adj_)
            component_densities_pullback!(nothing, m, point, ρ_adj, m_adj)
        end
    end
end

function compute_adjoint_source!(Q, intensities_, x_rays, els, number_of_atoms, absorption_coefficients, E)
    find_el_idx(x_ray) = findall(x -> x == element(x_ray), els)[1]
    el_idx = find_el_idx.(x_rays)

    σ = emissioncrosssection.([E], x_rays)

    # compute_(emiss, abs_coeff, number_of_atoms, intensities_) = emiss .* abs_coeff .* number_of_atoms .* intensities_
    # Q.eee[:, :, :] .= -sum.(compute_.([σ], absorption_coefficients, index_number_of_atoms.(number_of_atoms), [intensities_]))

    for (i, Q_) in enumerate(Q.eee)
        Q_[1] = 0.
        for (j, x_ray) in enumerate(x_rays)
            Q_[1] -= σ[j] * absorption_coefficients[i][j] * number_of_atoms[i][el_idx[j]] * intensities_[j]
        end
    end
end

function integrate_step_and_add!(intensities, gspec, x_rays, els, u, number_of_atoms, absorption_coefficients, E, ΔE, int_fac)
    dvol_ = dvol(gspec)

    find_el_idx(x_ray) = findall(x -> x == element(x_ray), els)[1]
    el_idx = find_el_idx.(x_rays)

    integral = zeros(length(x_rays))

    σ = emissioncrosssection.(Ref(E), x_rays)
    for (i, _) in enumerate(absorption_coefficients)
        for (r, _) in enumerate(x_rays)
            integral[r] += absorption_coefficients[i][r]*number_of_atoms[i][el_idx[r]]*u.eee[i][1]
        end
    end

    for (r, x_ray) in enumerate(x_rays)
        intensities[r] += ΔE * dvol_ * σ[r] * integral[r] * int_fac
    end
    return intensities
end

function integrate_step_and_add_pullback!(intensities, gspec, x_rays, els, u, number_of_atoms, absorption_coefficients, E, ΔE, int_fac, intensities_, number_of_atoms_, absorption_coefficients_)
    dvol_ = dvol(gspec)

    find_el_idx(x_ray) = findall(x -> x == element(x_ray), els)[1]
    el_idx = find_el_idx.(x_rays)

    integral = zeros(length(x_rays))

    σ = emissioncrosssection.(Ref(E), x_rays)
    # for (i, _) in enumerate(absorption_coefficients)
    #     for (r, _) in enumerate(x_rays)
    #         integral[r] += absorption_coefficients[i][r]*number_of_atoms[i][el_idx[r]]*u.eee[i][1]
    #     end
    # end

    # for (r, x_ray) in enumerate(x_rays)
    #     intensities[r] += ΔE * dvol * σ[r] * integral[r] * int_fac
    # end

    integral_ = zeros(length(x_rays))
    for (r, _) in enumerate(x_rays)
        integral_[r] += ΔE * dvol_ * σ[r] * int_fac * intensities_[r]
    end

    for (i, _) in enumerate(absorption_coefficients)
        for (r, _) in enumerate(x_rays)
            absorption_coefficients_[i][r] += number_of_atoms[i][el_idx[r]]*u.eee[i][1]*integral_[r]
            number_of_atoms_[i][el_idx[r]] += absorption_coefficients[i][r]*u.eee[i][1]*integral_[r]
        end
    end
end

## COMPUTE X_RAY GENERATION FIELD
function integrate_step_and_add_x_ray_generation!(x_ray_generation, x_rays, u, E, ΔE, int_fac)
    σ = emissioncrosssection.(Ref(E), x_rays)
    @inbounds for (i, _) in enumerate(u.eee)
        for (r, _) in enumerate(x_rays)
            x_ray_generation[i][r] += u.eee[i][1] * ΔE * σ[r] * int_fac
        end
    end
    return x_ray_generation
end
