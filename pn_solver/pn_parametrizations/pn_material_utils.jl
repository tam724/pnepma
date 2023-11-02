
density(m::PNMaterial, x::AbstractVector) = sum(component_densities(m, x))

function component_densities(m::PNMaterial{NE, T}, x::AbstractVector) where {NE, T}
    ρ = zeros(T, n_elements(m))
    component_densities!(ρ, m, x)
    return ρ
end

@adjoint function component_densities(m::PNMaterial{NE, T}, x::AbstractVector) where {NE, T}
    ρ = zeros(T, n_elements(m))
    component_densities!(ρ, m, x)
    function component_densities_pullback(ρ_)
        m_ = zero(m)
        component_densities_pullback!(nothing, m, x, ρ_, m_)
        return to_named_tuple(m_), nothing
    end
    return ρ, component_densities_pullback
end

function component_density(m::PNMaterial, x::AbstractVector, el_idx)
    @assert el_idx <= n_elements(m)
    ρ = zeros(n_elements(m))
    component_densities!(ρ, m, x)
    return ρ[el_idx]
end

## comparison functions

for MT in [:MassConcentrationPNMaterial, VolumeFractionPNMaterial, VolumeFractionPNMaterialSC, LinearDensityPNMaterial, LinearDensityPNMaterialSC, MassFractionPNMaterial, MassFractionPNMaterialSC]
    @eval begin
        function Zygote_hessian(f, m::$MT)
            p = param_vec(m)
            return Zygote.hessian(p -> f($MT(m, from_param_vec(m.p, p))), p)
        end

        function ForwardDiff_jacobian(f, m::$MT)
            p = param_vec(m)
            return ForwardDiff.jacobian(p -> f($MT(m, from_param_vec(m.p, p))), p)
        end

        function ForwardDiff_gradient(f, m::$MT)
            p = param_vec(m)
            return ForwardDiff.gradient(p -> f($MT(m, from_param_vec(m.p, p))), p)
        end

        function ForwardDiff_hessian(f, m::$MT)
            p = param_vec(m)
            return ForwardDiff.hessian(p -> f($MT(m, from_param_vec(m.p, p))), p)
        end
    end
end

function jacobian_finite_differences(f, m)
    new_m = zero(m)
    p = zeros(n_params(new_m))
    p = param_vec!(p, m)
    p = deepcopy(p)
    from_param_vec!(new_m, p)
    
    return FiniteDifferences.jacobian(central_fdm(2, 1), p -> f(from_param_vec!(new_m, p)), p)[1]
end

function diag_hessian_finite_differences(f, m)
    new_m = deepcopy(m)
    p = zeros(n_params(new_m))
    p = param_vec!(p, new_m)
    
    return FiniteDifferences.jacobian(central_fdm(3, 2), p -> f(from_param_vec!(new_m, p)), p)[1]
end

## some convenience functions
if @isdefined GridSpec
    function component_densities(m::PNMaterial, gspec::GridSpec{T}) where T
        n = n_elements(m)
        ρ = zeros(MStaggeredGridVariable{n, n, n, n, n, n, n, n, T}, gspec)
        for eo in all_eos()
            for (ρ_, x) in zip(ρ[eo], points(eo, gspec))
                component_densities!(ρ_, m, collect(x))
            end
        end
        return ρ
    end

    function component_densities_pullback(m::PNMaterial, gspec::GridSpec, ρ_, m_)
        for eo in all_eos()
            for (ρ__, x) in zip(ρ_[eo], points(eo, gspec))
                component_densities_pullback!(nothing, m, collect(x), ρ__, m_)
            end
        end
    end

    function density_eee(m::PNMaterial, gspec::GridSpec{T}) where T
        eee = EvenOddClassification{Even, Even, Even}
        return density.(Ref(m), collect.(points(eee, gspec)))
    end

    function component_densities_eee!(ρ, m::PNMaterial, gspec::GridSpec{T}) where T
        eee = EvenOddClassification{Even, Even, Even}
        for (i, x) in enumerate(points(eee, gspec))
            component_densities!(ρ[i], m, collect(x))
        end
    end

    function component_densities_eee(m::PNMaterial{NE, T}, gspec::GridSpec{U}) where {NE, T, U}
        n = n_elements(m)
        eee = EvenOddClassification{Even, Even, Even}
        ρ = [zeros(T, n) for _ in points(eee, gspec)]
        component_densities_eee!(ρ, m, gspec)
        return ρ
    end

    @adjoint function component_densities_eee(m::PNMaterial, gspec::GridSpec{T}) where T
        n = n_elements(m)
        eee = EvenOddClassification{Even, Even, Even}
        ρ = [zeros(n) for _ in points(eee, gspec)]
        for (ρ_, x) in zip(ρ, points(eee, gspec))
            component_densities!(ρ_, m, collect(x))
        end
        function component_densities_eee_pullback(ρ_)
            m_ = zero(m)
            eee = EvenOddClassification{Even, Even, Even}
            for (ρ__, x) in zip(ρ_, points(eee, gspec))
                component_densities_pullback!(nothing, m, collect(x), ρ__, m_)
            end
            return (m_, nothing)
        end
        return ρ, component_densities_eee_pullback
    end

    function component_densities_eee_pullback(m::PNMaterial, gspec::GridSpec, ρ_, m_)
        eee = EvenOddClassification{Even, Even, Even}
        for (ρ__, x) in zip(ρ_, points(eee, gspec))
            component_densities_pullback!(nothing, m, collect(x), ρ__, m_)
        end
    end


# PLOTTING FUNCTIONS
    for style in (:contourf, :surface, :heatmap)
        @eval begin
            function $style(m::PNMaterial, gspec::GridSpec, el_idx::Int)
                eee = EvenOddClassification{Even, Even, Even}
                ρ = component_density.(Ref(m), collect.(points(eee, gspec)), Ref(el_idx))
                Plots.plot(ρ[:, :, 1], st=Symbol($style))
            end

            function $style(m::PNMaterial, gspec::GridSpec)
                eee = EvenOddClassification{Even, Even, Even}
                ρ = density.(Ref(m), collect.(points(eee, gspec)))
                Plots.plot(ρ[:, :, 1], st=Symbol($style))
            end
        end
    end

        # function heatmap(m::PNMaterial, gspec::GridSpec, el_idx::Int)
        #     eee = EvenOddClassification{Even, Even, Even}
        #     ρ = component_density.(Ref(m), collect.(points(eee, gspec)), Ref(el_idx))
        #     Plots.plot(ρ[:, :, 1], st=:heatmap)
        # end

        # function surface(m::PNMaterial, gspec::GridSpec, el_idx::Int)
        #     eee = EvenOddClassification{Even, Even, Even}
        #     ρ = component_density.(Ref(m), collect.(points(eee, gspec)), Ref(el_idx))
        #     Plots.plot(ρ[:, :, 1], st=:surface)
        # end

    function get_index(dimpair)
        if is_non_singleton(dimpair)
            return Colon()
        else
            return 1
        end
    end

    function plot(m::PNMaterial, gspec::GridSpec, el_idx::Int, st=nothing)
        n_non_singleton = sum(is_non_singleton.(gspec.dims))
        if !isnothing(st)
            if n_non_singleton == 1
                @assert st == :line
            elseif n_non_singleton == 2
                @assert st == :contourf || st == :contour || st == :surface
            elseif n_non_singleton == 3
                #@assert st == 
            end
        end
        ρ = component_densities_eee(m, gspec)[get_index.(gspec.dims)...]
        if !isnothing(st)
            Plots.plot(getindex.(ρ, el_idx), st=st)
        else
            Plots.plot(getindex.(ρ, el_idx))
        end
    end

    function plot!(m::PNMaterial, gspec::GridSpec, el_idx::Int, st=nothing)
        n_non_singleton = sum(is_non_singleton.(gspec.dims))
        if !isnothing(st)
            if n_non_singleton == 1
                @assert st == :line
            elseif n_non_singleton == 2
                @assert st == :contourf || st == :contour || st == :surface
            elseif n_non_singleton == 3
                #@assert st == 
            end
        end
        ρ = component_densities_eee(m, gspec)[get_index.(gspec.dims)...]
        if !isnothing(st)
            Plots.plot!(getindex.(ρ, el_idx), st=st)
        else
            Plots.plot!(getindex.(ρ, el_idx))
        end
    end


    function is_non_singleton(dimpair)
        return !(typeof(dimpair.e) <: StaRMAP.SingletonDimension)
    end
end