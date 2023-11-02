
function compute_dt_max(problem::PNProblem, cfl)
    dx = step(grid(problem)[X])
    dy = step(grid(problem)[Y])
    dz = step(grid(problem)[Z])
    minS = minimum([minimum(minimum.(problem.S[eo])) for eo in all_eos()])
    return cfl/(1. /(dx*minS) + 1. /(dy*minS) + 1. /(dz*minS))
end

t_to_E(p::AdjointPNProblem, t) = E_cutoff(p) + t
E_to_t(p::AdjointPNProblem, E) = E - E_cutoff(p)

t_to_E(p::ForwardPNProblem, t) = E_initial(p) - t
E_to_t(p::ForwardPNProblem, E) = E_initial(p) - E

function step_pn!(problem::ForwardPNProblem, ρ, u, du, E, ΔE)
    t = E_to_t(problem, E)
    update_problem!(problem, ρ, E - ΔE/2, ΔE)
    t = step_pde!(problem, u, du, t, ΔE)
    E = t_to_E(problem, t)
    return E
end

function step_pn!(problem::AdjointPNProblem, ρ, u, du, E, ΔE)
    t = E_to_t(problem, E)
    update_problem!(problem, ρ, E + ΔE/2, ΔE)
    t = step_pde!(problem, u, du, t, ΔE)
    E = t_to_E(problem, t)
    return E
end

function evolve(problem::ForwardPNProblem, ρ, u, du)
    ΔE = (E_initial(problem) - E_cutoff(problem))/N_steps(problem)
    E = E_initial(problem)
    for i in 1:N_steps
        E = step_pn!(problem, ρ, u, du, E, ΔE)
    end
    return E
end

function evolve(problem::AdjointPNProblem, ρ, u, du)
    ΔE = (E_initial(problem) - E_cutoff(problem))/N_steps(problem)
    E = E_initial(problem)
    for i in 1:N_steps(problem)
        E = step_pn!(problem, ρ, u, du, E, ΔE)
    end
    return E
end

function animated_plot(problem::ForwardPNProblem{N, T}, m, type=:contourf) where {N, T}
    gspec = grid(problem)
    u = zeros(MPNSolverVariable{N, T}, gspec)
    du = zeros(MPNSolverVariable{N, T}, gspec)
    ρ = component_densities(m, gspec)
    ΔE = (E_initial(problem) - E_cutoff(problem))/N_steps(problem)
    # @show ΔE
    E = E_initial(problem)

    @gif for i in 1:N_steps(problem)
        E = step_pn!(problem, ρ, u, du, E, ΔE)
        # @show compute_dt_max(problem, 1.0), ΔE, E
        #plot([plot(u.eee[i, :, 1, 1], title=string(u.moments.eee[i])) for i in 1:9]..., layout=(3, 3))
        p = Plots.plot(points(gspec.dims[2].e) ./ 1e-7 , points(gspec.dims[1].e) ./ 1e-7, getindex.(dropdims(u.eee, dims = tuple(findall(size(u.eee) .== 1)...)), 1), st=type, aspect_ratio=:equal)
        Plots.title!(string(E))

        if i%10 == 0
            display(p)
        end
    end
end

function compute_and_save(problem::ForwardPNProblem{N, T}, m) where {N, T}
    gspec = grid(problem)
    u = zeros(MPNSolverVariable{N, T}, gspec)
    du = zeros(MPNSolverVariable{N, T}, gspec)
    ρ = component_densities(m, gspec)
    ΔE = (E_initial(problem) - E_cutoff(problem))/N_steps(problem)
    # @show ΔE
    E = E_initial(problem)

    u_store = []

    for i in 1:N_steps(problem)
        E = step_pn!(problem, ρ, u, du, E, ΔE)
        # @show compute_dt_max(problem, 1.0), ΔE, E
        #plot([plot(u.eee[i, :, 1, 1], title=string(u.moments.eee[i])) for i in 1:9]..., layout=(3, 3))
        # p = Plots.plot(points(gspec.dims[2].e) ./ 1e-7 , points(gspec.dims[1].e) ./ 1e-7, getindex.(dropdims(u.eee, dims = tuple(findall(size(u.eee) .== 1)...)), 1), st=:heatmap, aspect_ratio=:equal)
        # Plots.title!(string(E))

        push!(u_store, (E, deepcopy(getindex.(u.eee, 1))))
        # if i%10 == 0
        #     display(p)
        # end
    end
    return u_store
end

function animated_plot(problem::AdjointPNProblem{N, T}, m, x_rays, detector_position, beam_position, type=:contourf) where {N, T}
    λ = zeros(MPNSolverVariable{N, T}, gspec)
    dλ = zeros(MPNSolverVariable{N, T}, gspec)
    ρ = component_densities(m, gspec)
    ΔE = (E_initial(problem) - E_cutoff(problem))/N_steps(problem)
    E = E_cutoff(problem)

    ACS = compute_absorption_coefficients(m, x_rays, grid(problem), detector_position, beam_position.μ)
    NOA = compute_number_of_atoms(m, gspec)

    @gif for i in 1:N_steps(problem)
        compute_adjoint_source!(problem.Q, [1. for _ in x_rays], x_rays, m.elms, NOA, ACS, E)
        E = step_pn!(problem, ρ, λ, dλ, E, ΔE)
        # @show E
        #plot([plot(u.u.eee[i, :, 1, 1], title=string(u.moments.eee[i])) for i in 1:9]..., layout=(3, 3))
        plot(getindex.(dropdims(λ.eee, dims = tuple(findall(size(λ.eee) .== 1)...)), 1), st=type)
    end
end

function visualize2D(f, m)
    p = deepcopy(param_vec!(zeros(n_params(m)), m))
    RES = zeros(10, 10)
    for (i, x) in enumerate(range(p[1]-0.1, p[1]+0.1,length=10))
        for (j, y) in enumerate(range(p[2]-0.1, p[2]+0.1, length=10))
            m_temp = from_param_vec!(zero(m), [x, y])
            RES[i, j] = f(m_temp)
        end
    end
    plot(RES, st=:contourf)
end

function integrate_adjoint_densities!(u, du, λ, s, c, ΔE, dvol_, ρ_, int_fac, n_e, ::Type{EvenOddClassification{EOX, EOY, EOZ}}) where {EOX, EOY, EOZ}
    (n_x, n_y, n_z) = size(du)
    n_m = length(first(du))
    for i in StaRMAP.eo_range(EOX, n_x), j in StaRMAP.eo_range(EOY, n_y), k in StaRMAP.eo_range(EOZ, n_z)
        for e in 1:n_e
            for m in 1:n_m
                # ρ has dimensions [[n_elements], [n_x, n_y, n_z]]
                # ρ_[eo][i, j, k][e] += λ[eo][i, j, k][m] * (c[eo][e][m] * u[eo][i, j, k][m])*dvol_*ΔE*int_fac
                ρ_[i, j, k][e] += λ[i, j, k][m] * (s[e][m] * du[i, j, k][m] + c[e][m] * u[i, j, k][m])*dvol_*ΔE*int_fac
            end
        end
    end
end

function integrate_adjoint_densities!(problem, ρ, u, du, λ, E, ΔE, ρ_, int_fac)
    update_problem!(problem, ρ, E, ΔE)
    StaRMAP.apply_boundary!(StaRMAP.EvenPairity, problem, u)
    StaRMAP.apply_boundary!(StaRMAP.OddPairity, problem, u)
    StaRMAP.calc_du!(StaRMAP.EvenPairity, problem, du, u, ΔE, false)
    StaRMAP.calc_du!(StaRMAP.OddPairity, problem, du, u, ΔE, false)
    
    dvol_ = dvol(grid(problem))

    n_e = length(problem.elements)
    s = problem.s
    c = problem.c

    integrate_adjoint_densities!(u.eee, du.eee, λ.eee, s.eee, c.eee, ΔE, dvol_, ρ_.eee, int_fac, n_e, EvenOddClassification{Even, Even, Even})
    integrate_adjoint_densities!(u.eeo, du.eeo, λ.eeo, s.eeo, c.eeo, ΔE, dvol_, ρ_.eeo, int_fac, n_e, EvenOddClassification{Even, Even, Odd})
    integrate_adjoint_densities!(u.eoe, du.eoe, λ.eoe, s.eoe, c.eoe, ΔE, dvol_, ρ_.eoe, int_fac, n_e, EvenOddClassification{Even, Odd, Even})
    integrate_adjoint_densities!(u.eoo, du.eoo, λ.eoo, s.eoo, c.eoo, ΔE, dvol_, ρ_.eoo, int_fac, n_e, EvenOddClassification{Even, Odd, Odd})

    integrate_adjoint_densities!(u.oee, du.oee, λ.oee, s.oee, c.oee, ΔE, dvol_, ρ_.oee, int_fac, n_e, EvenOddClassification{Odd, Even, Even})
    integrate_adjoint_densities!(u.oeo, du.oeo, λ.oeo, s.oeo, c.oeo, ΔE, dvol_, ρ_.oeo, int_fac, n_e, EvenOddClassification{Odd, Even, Odd})
    integrate_adjoint_densities!(u.ooe, du.ooe, λ.ooe, s.ooe, c.ooe, ΔE, dvol_, ρ_.ooe, int_fac, n_e, EvenOddClassification{Odd, Odd, Even})
    integrate_adjoint_densities!(u.ooo, du.ooo, λ.ooo, s.ooo, c.ooo, ΔE, dvol_, ρ_.ooo, int_fac, n_e, EvenOddClassification{Odd, Odd, Odd})

    # plot(u.u.eee[1, :, 1, 1])
    # display(plot!(du.u.eee[1, :, 1, 1]))
    # @show du

    # p = plot(getindex.(u.eee[:, 1, 1], 1))
    # p = plot!(getindex.(du.eee[:, 1, 1], 1))
    # display(p)

    # for eo in all_eos()
    #     (n_x, n_y, n_z) = size(du[eo])
    #     n_m = length(first(du[eo]))
    #     for i in StaRMAP.eo_range(eo_in(eo, X), n_x), j in StaRMAP.eo_range(eo_in(eo, Y), n_y), k in StaRMAP.eo_range(eo_in(eo, Z), n_z)
    #         for e in 1:n_e
    #             for m in 1:n_m
    #                 # ρ has dimensions [[n_elements], [n_x, n_y, n_z]]
    #                 # ρ_[eo][i, j, k][e] += λ[eo][i, j, k][m] * (c[eo][e][m] * u[eo][i, j, k][m])*dvol_*ΔE*int_fac
    #                 ρ_[eo][i, j, k][e] += λ[eo][i, j, k][m] * (s[eo][e][m] * du[eo][i, j, k][m] + c[eo][e][m] * u[eo][i, j, k][m])*dvol_*ΔE*int_fac
    #             end
    #         end
    #     end
    # end
end

function calc_x_ray_generation_field(problem::ForwardPNProblem{N, T}, m, x_rays) where {N, T}
    u = zeros(MPNSolverVariable{N, T}, grid(problem))
    du = zeros(MPNSolverVariable{N, T}, grid(problem))
    E = E_initial(problem)
    ΔE = (E_initial(problem) - E_cutoff(problem))/N_steps(problem)

    ρ = component_densities(m, grid(problem))

    x_ray_generation = [zeros(length(x_rays)) for _ in u.eee]

    # @show E
    x_ray_generation = integrate_step_and_add_x_ray_generation!(x_ray_generation, x_rays, u, E, ΔE, 0.5)

    for i in 1:N_steps(problem)
        t = E_to_t(problem, E)
        update_problem!(problem, ρ, E - ΔE/2, ΔE)
        t = step_pde!(problem, u, du, t, ΔE)
        E = t_to_E(problem, t)
        # @show compute_dt_max(problem, 1.0), ΔE, E
        int_fac = i == N_steps ? 0.5 : 1.0
        x_ray_generation = integrate_step_and_add_x_ray_generation!(x_ray_generation, x_rays, u, E, ΔE, int_fac)
    end
    return x_ray_generation
end

function calc_intensities_from_x_ray_generation(gspec::GridSpec, m, x_rays, detector_position, beam_position, x_ray_generation) where {N, T}
    number_of_atoms = compute_number_of_atoms(m, gspec)
    absorption_coefficients = compute_absorption_coefficients(m, x_rays, gspec, detector_position, beam_position.μ)

    dvol_ = dvol(gspec)

    find_el_idx(x_ray) = findall(x -> x == element(x_ray), m.elms)[1]
    el_idx = find_el_idx.(x_rays)

    intensities = zeros(length(x_rays))

    for (i, _) in enumerate(absorption_coefficients)
        for (r, _) in enumerate(x_rays)
            intensities[r] += absorption_coefficients[i][r]*number_of_atoms[i][el_idx[r]]*x_ray_generation[i][r] * dvol_
        end
    end
    return intensities
end

@adjoint function calc_intensities_from_x_ray_generation(gspec::GridSpec, m, x_rays, detector_position, beam_position, x_ray_generation) where {N, T}
    number_of_atoms = compute_number_of_atoms(m, gspec)
    absorption_coefficients = compute_absorption_coefficients(m, x_rays, gspec, detector_position, beam_position.μ)

    dvol_ = dvol(gspec)

    find_el_idx(x_ray) = findall(x -> x == element(x_ray), m.elms)[1]
    el_idx = find_el_idx.(x_rays)

    intensities = zeros(length(x_rays))

    for (i, _) in enumerate(absorption_coefficients)
        for (r, _) in enumerate(x_rays)
            intensities[r] += absorption_coefficients[i][r]*number_of_atoms[i][el_idx[r]]*x_ray_generation[i][r] * dvol_
        end
    end
    function calc_intensities_from_x_ray_generation_pullback(intensities_)
        number_of_atoms_ = [zeros(length(m.elms)) for _ in number_of_atoms]
        absorption_coefficients_ = [zeros(length(x_rays)) for _ in absorption_coefficients]
        for (i, _) in enumerate(absorption_coefficients)
            for (r, _) in enumerate(x_rays)
                absorption_coefficients_[i][r] += number_of_atoms[i][el_idx[r]]*x_ray_generation[i][r] * dvol_ * intensities_[r]
                number_of_atoms_[i][el_idx[r]] += absorption_coefficients[i][r]*x_ray_generation[i][r] * dvol_ * intensities_[r]
            end
        end

        m_ = zero(m)
        compute_absorption_coefficient_pullback(m, x_rays, gspec, detector_position, beam_position.μ, absorption_coefficients, m_, absorption_coefficients_)
        compute_number_of_atoms_pullback(m, gspec, m_, number_of_atoms_)
        return (nothing, to_named_tuple(m_), nothing, nothing, nothing, nothing)
    end
    return intensities, calc_intensities_from_x_ray_generation_pullback
end


function calc_intensities_from_x_ray_generation_no_absorption(gspec::GridSpec, m, AM, x_rays, number_of_atoms, x_ray_generation) where {N, T}
    number_of_atoms = compute_number_of_atoms!(number_of_atoms, AM, m, gspec)

    dvol_ = dvol(gspec)

    find_el_idx(x_ray) = findall(x -> x == element(x_ray), m.elms)[1]
    el_idx = find_el_idx.(x_rays)

    intensities = zeros(length(x_rays))

    @inbounds for (i, _) in enumerate(x_ray_generation)
        for (r, _) in enumerate(x_rays)
            intensities[r] += number_of_atoms[i][el_idx[r]]*x_ray_generation[i][r] * dvol_
        end
    end
    return intensities
end

@adjoint function calc_intensities_from_x_ray_generation_no_absorption(gspec::GridSpec, m, AM, x_rays, number_of_atoms, x_ray_generation) where {N, T}
    number_of_atoms = compute_number_of_atoms!(number_of_atoms, AM, m, gspec)

    dvol_ = dvol(gspec)

    find_el_idx(x_ray) = findall(x -> x == element(x_ray), m.elms)[1]
    el_idx = find_el_idx.(x_rays)

    intensities = zeros(length(x_rays))

    @inbounds for (i, _) in enumerate(x_ray_generation)
        for (r, _) in enumerate(x_rays)
            intensities[r] += number_of_atoms[i][el_idx[r]]*x_ray_generation[i][r] * dvol_
        end
    end
    function calc_intensities_from_x_ray_generation_pullback(intensities_)
        # careful, big hack !
        eee = EvenOddClassification{Even, Even, Even}
        number_of_atoms_ = zeros(length(m.elms))
        m_ = zero(m)
        x_ = @MVector [0., 0., 0.]

        @inbounds for (i, point) in enumerate(points(eee, gspec))
            number_of_atoms_ .= 0.
            collect!(x_, point)
            for r in 1:length(x_rays)
                number_of_atoms_[el_idx[r]] += x_ray_generation[i][r] * dvol_ * intensities_[r]
            end
            number_of_atoms_ ./= AM
            component_densities_pullback!(nothing, m, x_, number_of_atoms_, m_)
        end

        # compute_number_of_atoms!_pullback(m, AM, gspec, m_, number_of_atoms_)
        return (nothing, to_named_tuple(m_), nothing, nothing, nothing, nothing, nothing)
    end
    return intensities, calc_intensities_from_x_ray_generation_pullback
end


function calc_intensities(problem::ForwardPNProblem{N, T}, m, x_rays, detector_position) where {N, T}
    u = zeros(MPNSolverVariable{N, T}, grid(problem))
    du = zeros(MPNSolverVariable{N, T}, grid(problem))
    E = E_initial(problem)
    ΔE = (E_initial(problem) - E_cutoff(problem))/N_steps(problem)

    ρ = component_densities(m, grid(problem))

    number_of_atoms = compute_number_of_atoms(m, grid(problem))
    absorption_coefficients = compute_absorption_coefficients(m, x_rays, grid(problem), detector_position, problem.beam_position.μ)

    intensities = zeros(length(x_rays))

    # @show E
    intensities = integrate_step_and_add!(intensities, grid(problem), x_rays, m.elms, u, number_of_atoms, absorption_coefficients, E, ΔE, 0.5)

    for i in 1:N_steps(problem)
        t = E_to_t(problem, E)
        update_problem!(problem, ρ, E - ΔE/2, ΔE)
        t = step_pde!(problem, u, du, t, ΔE)
        E = t_to_E(problem, t)
        # @show compute_dt_max(problem, 1.0), ΔE, E
        int_fac = i == N_steps ? 0.5 : 1.0
        integrate_step_and_add!(intensities, grid(problem), x_rays, m.elms, u, number_of_atoms, absorption_coefficients, E, ΔE, int_fac)
    end
    return intensities
end

Zygote.@adjoint function calc_intensities(problem::ForwardPNProblem{N, T}, m, x_rays, detector_position) where {N, T}
    u = zeros(MPNSolverVariable{N, T}, grid(problem))
    du = zeros(MPNSolverVariable{N, T}, grid(problem))
    E = E_initial(problem)
    ΔE = (E_initial(problem) - E_cutoff(problem))/N_steps(problem)

    ρ = component_densities(m, grid(problem)) # do not ignore this

    number_of_atoms = compute_number_of_atoms(m, grid(problem))
    absorption_coefficients = compute_absorption_coefficients(m, x_rays, grid(problem), detector_position, problem.beam_position.μ)

    intensities = zeros(length(x_rays))

    u_tape = []
    push!(u_tape, deepcopy(u))
    # @show E
    intensities = integrate_step_and_add!(intensities, grid(problem), x_rays, m.elms, u, number_of_atoms, absorption_coefficients, E, ΔE, 0.5)

    for i in 1:N_steps(problem)
        t = E_to_t(problem, E)
        update_problem!(problem, ρ, E - ΔE/2, ΔE)
        t = step_pde!(problem, u, du, t, ΔE)
        E = t_to_E(problem, t)
        # @show E
        int_fac = i == N_steps ? 0.5 : 1.0
        push!(u_tape, deepcopy(u))
        integrate_step_and_add!(intensities, grid(problem), x_rays, m.elms, u, number_of_atoms, absorption_coefficients, E, ΔE, int_fac)
    end
    function calc_intensities_pullback(intensities_)
        adjoint_problem = AdjointPNProblem(problem)
        λ = zeros(MPNSolverVariable{N, T}, gspec)
        dλ = zeros(MPNSolverVariable{N, T}, gspec)
        
        number_of_atoms_ = [zeros(length(m.elms)) for _ in number_of_atoms]
        absorption_coefficients_ = [zeros(length(x_rays)) for _ in absorption_coefficients]
        n = n_elements(m)
        ρ_ = zeros(MStaggeredGridVariable{n, n, n, n, n, n, n, n, T}, gspec)
        E = E_cutoff(adjoint_problem)
        for i in reverse(1:N_steps(problem))
            u = u_tape[i+1]
            int_fac = i == N_steps ? 0.5 : 1.0
            integrate_adjoint_densities!(problem, ρ, u, du, λ, E, ΔE, ρ_, int_fac)
            integrate_step_and_add_pullback!(nothing, grid(problem), x_rays, m.elms, u, number_of_atoms, absorption_coefficients, E, ΔE, int_fac, intensities_, number_of_atoms_, absorption_coefficients_)
            t = E_to_t(adjoint_problem, E)
            compute_adjoint_source!(adjoint_problem.Q, intensities_, x_rays, m.elms, number_of_atoms, absorption_coefficients, E + ΔE/2.)            
            update_problem!(adjoint_problem, ρ, E + ΔE/2, ΔE)
            t = step_pde!(adjoint_problem, λ, dλ, t, ΔE)
            E = t_to_E(adjoint_problem, t)
        end
        u = u_tape[1]
        integrate_adjoint_densities!(problem, ρ, u, du, λ, E, ΔE, ρ_, 0.5)
        integrate_step_and_add_pullback!(nothing, grid(problem), x_rays, m.elms, u, number_of_atoms, absorption_coefficients, E, ΔE, 0.5, intensities_, number_of_atoms_, absorption_coefficients_)

        m_ = zero(m)
        component_densities_pullback(m, gspec, ρ_, m_)
        compute_absorption_coefficient_pullback(m, x_rays, grid(problem), detector_position, problem.beam_position.μ, absorption_coefficients, m_, absorption_coefficients_)
        compute_number_of_atoms_pullback(m, grid(problem), m_, number_of_atoms_)
        return (nothing, to_named_tuple(m_), nothing, nothing)
    end
    return intensities, calc_intensities_pullback
end