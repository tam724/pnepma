abstract type PNProblem{N, T} <: StaRMAPProblem end

struct ForwardPNProblem{N, T, NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO} <: PNProblem{N, T}
    moments::Moments
    elements::Vector{Element}
    #Ax::StaRMAP.SAdvectionMatrixX{NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T}
    #Ay::StaRMAP.SAdvectionMatrixY{NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T}
    #Az::StaRMAP.SAdvectionMatrixZ{NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T}

    Ax::StaRMAP.AdvectionMatrixCSC{T, Int64}
    Ay::StaRMAP.AdvectionMatrixCSC{T, Int64}
    Az::StaRMAP.AdvectionMatrixCSC{T, Int64}

    Bx::BoundaryConditionSource
    By::BoundaryConditionZero
    Bz::BoundaryConditionZero

    s::StaRMAP.EvenOddProperty{Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}}
    c::StaRMAP.EvenOddProperty{Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}}

    # # S \partial_t u + sum A_i \partial_{x_i} u + C u = Q
    S::StaRMAP.MStaggeredGridVariable{NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T} # capacity coefficient (diagonal matrix, stored as a vector)
    C::StaRMAP.MStaggeredGridVariable{NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T} # transport coefficient (diagonal matrix, stored as a vector)
    Q::StaRMAP.ZerosStaggeredGridVariable{T} # source term

    gx_temp::StaRMAP.VBoundaryGridVariableX

    sp_funcs
    tc_funcs

    beam_energy
    beam_position
    beam_direction
    gspec::GridSpec{T}
    E_initial::T
    E_cutoff::T
    N_steps::Int64
end

struct AdjointPNProblem{N, T, NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO} <: PNProblem{N, T}
    moments::Moments
    elements::Vector{Element}
    #Ax::StaRMAP.SAdvectionMatrixX{NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T}
    #Ay::StaRMAP.SAdvectionMatrixY{NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T}
    #Az::StaRMAP.SAdvectionMatrixZ{NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T}

    Ax::StaRMAP.AdvectionMatrixCSC{T, Int64}
    Ay::StaRMAP.AdvectionMatrixCSC{T, Int64}
    Az::StaRMAP.AdvectionMatrixCSC{T, Int64}

    Bx::BoundaryConditionZero
    By::BoundaryConditionZero
    Bz::BoundaryConditionZero

    s::StaRMAP.EvenOddProperty{Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}}
    c::StaRMAP.EvenOddProperty{Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}}

    # # S \partial_t u + sum A_i \partial_{x_i} u + C u = Q
    S::StaRMAP.MStaggeredGridVariable{NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T} # capacity coefficient
    C::StaRMAP.MStaggeredGridVariable{NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T} # transport coefficient (diagonal matrix, stored as a vector)
    Q::StaRMAP.MStaggeredGridVariable{NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T} # source term

    sp_funcs
    tc_funcs

    gspec::GridSpec{T}
    E_initial
    E_cutoff
    N_steps
end

function ForwardPNProblem{N, T}(gspec::GridSpec, elements, beam_energy, beam_position, beam_direction, E_initial, E_cutoff, N_steps) where {N, T}
    moments = make_moments(N)
    problem = ForwardPNProblem{N, T, N_eee(N), N_eeo(N), N_eoe(N), N_eoo(N), N_oee(N), N_oeo(N), N_ooe(N), N_ooo(N)}(
        moments,
        elements,
        StaRMAP.AdvectionMatrixCSC{T, Int64}([make_advection_matrix(T, moments[eo_to], moments[switch_eo(eo_to, X)], X) for eo_to in all_eos()]...),
        StaRMAP.AdvectionMatrixCSC{T, Int64}([make_advection_matrix(T, moments[eo_to], moments[switch_eo(eo_to, Y)], Y) for eo_to in all_eos()]...),
        StaRMAP.AdvectionMatrixCSC{T, Int64}([make_advection_matrix(T, moments[eo_to], moments[switch_eo(eo_to, Z)], Z) for eo_to in all_eos()]...),
        #StaRMAP.SAdvectionMatrixX{N_eee(N), N_eeo(N), N_eoe(N), N_eoo(N), N_oee(N), N_oeo(N), N_ooe(N), N_ooo(N), T}(Array.([make_advection_matrix(T, moments[eo_to], moments[switch_eo(eo_to, X)], X) for eo_to in all_eos()])...),
        #StaRMAP.SAdvectionMatrixY{N_eee(N), N_eeo(N), N_eoe(N), N_eoo(N), N_oee(N), N_oeo(N), N_ooe(N), N_ooo(N), T}(Array.([make_advection_matrix(T, moments[eo_to], moments[switch_eo(eo_to, Y)], Y) for eo_to in all_eos()])...),
        #StaRMAP.SAdvectionMatrixZ{N_eee(N), N_eeo(N), N_eoe(N), N_eoo(N), N_oee(N), N_oeo(N), N_ooe(N), N_ooo(N), T}(Array.([make_advection_matrix(T, moments[eo_to], moments[switch_eo(eo_to, Z)], Z) for eo_to in all_eos()])...),
        BoundaryConditionSource(
            StaRMAP.BoundaryMatrixX{T}([make_boundary_matrix(T, moments[eo_to], moments[switch_eo(eo_to, X)], X) for eo_to in all_eos() if is_odd_in(eo_to, X)]...),
            zeros(StaRMAP.VBoundaryGridVariableX{Float64}, [N_oee(N), N_oeo(N), N_ooe(N), N_ooo(N)], gspec)),
        BoundaryConditionZero(StaRMAP.BoundaryMatrixY{T}([make_boundary_matrix(T, moments[eo_to], moments[switch_eo(eo_to, Y)], Y) for eo_to in all_eos() if is_odd_in(eo_to, Y)]...)),
        BoundaryConditionZero(StaRMAP.BoundaryMatrixZ{T}([make_boundary_matrix(T, moments[eo_to], moments[switch_eo(eo_to, Z)], Z) for eo_to in all_eos() if is_odd_in(eo_to, Z)]...)),
        StaRMAP.EvenOddProperty{Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}}([[zeros(T, length(moments[eo])) for _ in 1:length(elements)] for eo in all_eos()]...),
        StaRMAP.EvenOddProperty{Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}}([[zeros(T, length(moments[eo])) for _ in 1:length(elements)] for eo in all_eos()]...),
        zeros(StaRMAP.MStaggeredGridVariable{N_eee(N), N_eeo(N), N_eoe(N), N_eoo(N), N_oee(N), N_oeo(N), N_ooe(N), N_ooo(N), T}, gspec),
        zeros(StaRMAP.MStaggeredGridVariable{N_eee(N), N_eeo(N), N_eoe(N), N_eoo(N), N_oee(N), N_oeo(N), N_ooe(N), N_ooo(N), T}, gspec),
        zeros(StaRMAP.ZerosStaggeredGridVariable{T}, [N_eee(N), N_eeo(N), N_eoe(N), N_eoo(N), N_oee(N), N_oeo(N), N_ooe(N), N_ooo(N)], gspec),
        zeros(StaRMAP.VBoundaryGridVariableX{Float64}, [N_oee(N), N_oeo(N), N_ooe(N), N_ooo(N)], gspec),
        stopping_power_funcs(elements),
        transport_coefficient_funcs(elements, N),
        beam_energy, 
        beam_position,
        beam_direction,
        gspec,
        E_initial, 
        E_cutoff, 
        N_steps
    )
    setup_boundary_condition!(problem)
    return problem
end

function AdjointPNProblem{N, T}(gspec::GridSpec, elements, E_initial, E_cutoff, N_steps) where {N, T}
    moments = make_moments(N)
    problem = AdjointPNProblem{N, T, N_eee(N), N_eeo(N), N_eoe(N), N_eoo(N), N_oee(N), N_oeo(N), N_ooe(N), N_ooo(N)}(
        moments,
        elements,
        StaRMAP.AdvectionMatrixCSC{T, Int64}([-make_advection_matrix(T, moments[eo_to], moments[switch_eo(eo_to, X)], X) for eo_to in all_eos()]...),
        StaRMAP.AdvectionMatrixCSC{T, Int64}([-make_advection_matrix(T, moments[eo_to], moments[switch_eo(eo_to, Y)], Y) for eo_to in all_eos()]...),
        StaRMAP.AdvectionMatrixCSC{T, Int64}([-make_advection_matrix(T, moments[eo_to], moments[switch_eo(eo_to, Z)], Z) for eo_to in all_eos()]...),
        BoundaryConditionZero(StaRMAP.BoundaryMatrixX{T}([make_boundary_matrix_adjoint(T, moments[eo_to], moments[switch_eo(eo_to, X)], X) for eo_to in all_eos() if is_odd_in(eo_to, X)]...)),
        BoundaryConditionZero(StaRMAP.BoundaryMatrixY{T}([make_boundary_matrix_adjoint(T, moments[eo_to], moments[switch_eo(eo_to, Y)], Y) for eo_to in all_eos() if is_odd_in(eo_to, Y)]...)),
        BoundaryConditionZero(StaRMAP.BoundaryMatrixZ{T}([make_boundary_matrix_adjoint(T, moments[eo_to], moments[switch_eo(eo_to, Z)], Z) for eo_to in all_eos() if is_odd_in(eo_to, Z)]...)),
        StaRMAP.EvenOddProperty{Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}}([[zeros(T, length(moments[eo])) for _ in 1:length(elements)] for eo in all_eos()]...),
        StaRMAP.EvenOddProperty{Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}, Vector{Vector{T}}}([[zeros(T, length(moments[eo])) for _ in 1:length(elements)] for eo in all_eos()]...),
        zeros(StaRMAP.MStaggeredGridVariable{N_eee(N), N_eeo(N), N_eoe(N), N_eoo(N), N_oee(N), N_oeo(N), N_ooe(N), N_ooo(N), T}, gspec),
        zeros(StaRMAP.MStaggeredGridVariable{N_eee(N), N_eeo(N), N_eoe(N), N_eoo(N), N_oee(N), N_oeo(N), N_ooe(N), N_ooo(N), T}, gspec),
        zeros(StaRMAP.MStaggeredGridVariable{N_eee(N), N_eeo(N), N_eoe(N), N_eoo(N), N_oee(N), N_oeo(N), N_ooe(N), N_ooo(N), T}, gspec),
        stopping_power_funcs(elements),
        transport_coefficient_funcs(elements, N),
        gspec,
        E_initial,
        E_cutoff,
        N_steps
    )
    return problem
end

function AdjointPNProblem(frwd::ForwardPNProblem{N, T}) where {N, T}
    return AdjointPNProblem{N, T}(grid(frwd), frwd.elements, frwd.E_initial, frwd.E_cutoff, frwd.N_steps)
end

advection_matrix(p::PNProblem, ::Type{X}) = p.Ax
advection_matrix(p::PNProblem, ::Type{Y}) = p.Ay
advection_matrix(p::PNProblem, ::Type{Z}) = p.Az

boundary_condition(p::PNProblem, ::Type{X}) = p.Bx
boundary_condition(p::PNProblem, ::Type{Y}) = p.By
boundary_condition(p::PNProblem, ::Type{Z}) = p.Bz

capacity_coefficient(p::PNProblem) = p.S
homogeneous_term(p::PNProblem) = p.C
source_term(p::PNProblem) = p.Q

grid(p::PNProblem) = p.gspec

E_initial(p::PNProblem) = p.E_initial
E_cutoff(p::PNProblem) = p.E_cutoff

N_steps(p::PNProblem) = p.N_steps

function update_coefficients!(problem::ForwardPNProblem, E)
    for eo in all_eos()
        for (i, sp_func) in enumerate(problem.sp_funcs)
            problem.s[eo][i] .= sp_func(E)
        end
    end
    for i in 1:length(problem.elements)
        tc = problem.tc_funcs[i](E)
        ds = problem.sp_funcs[i]'(E)
        for eo in all_eos()
            problem.c[eo][i] .= -ds
            for (j, (degree, order)) in enumerate(problem.moments[eo])
                problem.c[eo][i][j] -= tc[degree + 1] - tc[1]
            end
        end
    end
end

function update_coefficients!(problem::AdjointPNProblem, E)
    for eo in all_eos()
        for (i, sp_func) in enumerate(problem.sp_funcs)
            problem.s[eo][i] .= sp_func(E)
        end
    end
    for i in 1:length(problem.elements)
        tc = problem.tc_funcs[i](E)
        for eo in all_eos()
            problem.c[eo][i] .= 0.
            for (j, (degree, order)) in enumerate(problem.moments[eo])
                problem.c[eo][i][j] -= tc[degree + 1] - tc[1]
            end
        end
    end
end

function setup_boundary_condition!(problem::ForwardPNProblem)
    gspec = grid(problem)
    for eo in all_eos_odd_in(X)
        mms = beam_moments(problem.moments[eo], problem.beam_direction)
        eo_eval = switch_eo(eo, X)
        pos_pdf(x) = pdf(problem.beam_position, collect(x))
        ppos = pos_pdf.(points(eo_eval, gspec)[1, :, :])
        for i in 1:size(eo, gspec, 2), j in 1:size(eo, gspec, 3)
            problem.gx_temp[eo][2, i, j] .= mms .* ppos[i, j]
        end
    end
end

function update_boundary_condition!(problem::ForwardPNProblem, E, ΔE)
    c = (cdf(problem.beam_energy, E + ΔE) - cdf(problem.beam_energy, E))/ΔE
    # c = pdf(problem.beam_energy, E)
    for eo in all_eos_odd_in(X)
        problem.Bx.g[eo] .= problem.gx_temp[eo] * c
    end
end

function weighted_sum!(S, ρ, s)
    for i in eachindex(S)
        S[i] .= 0.
        for e in 1:length(s)
            S[i] .+= ρ[i][e] .* s[e]
        end
    end
end

function update_constant!(S, c)
    for i in eachindex(S)
        S[i] .= c
    end
end

function update_problem!(problem::ForwardPNProblem, ρ, E, ΔE)
    update_coefficients!(problem, E)
    update_boundary_condition!(problem, E, ΔE)

    weighted_sum!(problem.S.eee, ρ.eee, problem.s.eee)
    weighted_sum!(problem.S.eeo, ρ.eeo, problem.s.eeo)
    weighted_sum!(problem.S.eoe, ρ.eoe, problem.s.eoe)
    weighted_sum!(problem.S.eoo, ρ.eoo, problem.s.eoo)
    weighted_sum!(problem.S.oee, ρ.oee, problem.s.oee)
    weighted_sum!(problem.S.oeo, ρ.oeo, problem.s.oeo)
    weighted_sum!(problem.S.ooe, ρ.ooe, problem.s.ooe)
    weighted_sum!(problem.S.ooo, ρ.ooo, problem.s.ooo)

    weighted_sum!(problem.C.eee, ρ.eee, problem.c.eee)
    weighted_sum!(problem.C.eeo, ρ.eeo, problem.c.eeo)
    weighted_sum!(problem.C.eoe, ρ.eoe, problem.c.eoe)
    weighted_sum!(problem.C.eoo, ρ.eoo, problem.c.eoo)
    weighted_sum!(problem.C.oee, ρ.oee, problem.c.oee)
    weighted_sum!(problem.C.oeo, ρ.oeo, problem.c.oeo)
    weighted_sum!(problem.C.ooe, ρ.ooe, problem.c.ooe)
    weighted_sum!(problem.C.ooo, ρ.ooo, problem.c.ooo)
end

function update_problem!(problem::AdjointPNProblem, ρ, E, ΔE)
    update_coefficients!(problem, E)

    weighted_sum!(problem.S.eee, ρ.eee, problem.s.eee)
    weighted_sum!(problem.S.eeo, ρ.eeo, problem.s.eeo)
    weighted_sum!(problem.S.eoe, ρ.eoe, problem.s.eoe)
    weighted_sum!(problem.S.eoo, ρ.eoo, problem.s.eoo)
    weighted_sum!(problem.S.oee, ρ.oee, problem.s.oee)
    weighted_sum!(problem.S.oeo, ρ.oeo, problem.s.oeo)
    weighted_sum!(problem.S.ooe, ρ.ooe, problem.s.ooe)
    weighted_sum!(problem.S.ooo, ρ.ooo, problem.s.ooo)

    weighted_sum!(problem.C.eee, ρ.eee, problem.c.eee)
    weighted_sum!(problem.C.eeo, ρ.eeo, problem.c.eeo)
    weighted_sum!(problem.C.eoe, ρ.eoe, problem.c.eoe)
    weighted_sum!(problem.C.eoo, ρ.eoo, problem.c.eoo)
    weighted_sum!(problem.C.oee, ρ.oee, problem.c.oee)
    weighted_sum!(problem.C.oeo, ρ.oeo, problem.c.oeo)
    weighted_sum!(problem.C.ooe, ρ.ooe, problem.c.ooe)
    weighted_sum!(problem.C.ooo, ρ.ooo, problem.c.ooo)
end