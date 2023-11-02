
N_eee(N) = Int(max(0, (1/2)*floor((N+2)/2)*floor((N+4)/2)))::Int
N_eeo(N) = N_eee(N-1)::Int
N_eoe(N) = N_eee(N-1)::Int
N_oee(N) = N_eee(N-1)::Int
N_eoo(N) = N_eee(N-2)::Int
N_oeo(N) = N_eee(N-2)::Int
N_ooe(N) = N_eee(N-2)::Int
N_ooo(N) = N_eee(N-3)::Int

Moments = StaRMAP.EvenOddProperty{Vector{Moment}, Vector{Moment}, Vector{Moment}, Vector{Moment}, Vector{Moment}, Vector{Moment}, Vector{Moment}, Vector{Moment}}
make_moments(Nmax::Int)::Moments = Moments([[m for m in all_moments(Nmax) if full_eo_classification(m) == eo] for eo in all_eos()]...)

const VPNSolverVariable{T} = StaRMAP.VStaggeredGridVariable{T}
const MPNSolverVariable{N, T, NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO} = StaRMAP.MStaggeredGridVariable{NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T}
const SPNSolverVariable{N, T, NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO} = StaRMAP.SStaggeredGridVariable{NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T}

import Base: zeros

##
function zeros(::Type{VPNSolverVariable{T}}, Nmax, gspec) where T
    return zeros(StaRMAP.VStaggeredGridVariable{T}, [N_eee(Nmax), N_eeo(Nmax), N_eoe(Nmax), N_eoo(Nmax), N_oee(Nmax), N_oeo(Nmax), N_ooe(Nmax), N_ooo(Nmax)], gspec)
end

function zeros(::Type{MPNSolverVariable{N, T}}, gspec) where {N, T}
    return zeros(StaRMAP.MStaggeredGridVariable{N_eee(N), N_eeo(N), N_eoe(N), N_eoo(N), N_oee(N), N_oeo(N), N_ooe(N), N_ooo(N), T}, gspec)
end

function zeros(::Type{SPNSolverVariable{N, T}}, gspec) where {N, T}
    return zeros(StaRMAP.SStaggeredGridVariable{N_eee(N), N_eeo(N), N_eoe(N), N_eoo(N), N_oee(N), N_oeo(N), N_ooe(N), N_ooo(N), T}, gspec)
end
##