function PNAdvectionMatrixCSC(::Type{T}, ::Val{N}, ::Type{D}) where {T, N, D<:StaRMAP.Dimension}
    moments = make_moments(N)
    @show moments
    return StaRMAP.AdvectionMatrixCSC{T, Int64}([make_advection_matrix(T, moments[eo_to], moments[switch_eo(eo_to, D)], D) for eo_to in all_eos()]...)
end