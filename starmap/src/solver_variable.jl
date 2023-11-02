import Base: zeros, ones

const StaggeredGridVariable{EEE, EEO, EOE, EOO, OEE, OEO, OOE, OOO} = EvenOddProperty{Array{EEE, 3}, Array{EEO, 3}, Array{EOE, 3}, Array{EOO, 3}, Array{OEE, 3}, Array{OEO, 3}, Array{OOE, 3}, Array{OOO, 3}}
const VStaggeredGridVariable{T} = StaggeredGridVariable{Vector{T}, Vector{T}, Vector{T}, Vector{T}, Vector{T}, Vector{T}, Vector{T}, Vector{T}}

function zeros(::Type{VStaggeredGridVariable{T}}, N::Vector{Int}, gspec::GridSpec) where T
    eee, eeo, eoe, eoo, oee, oeo, ooe, ooo = all_eos()
    return VStaggeredGridVariable{T}(
        [zeros(T, N[1]) for _ in 1:size(eee, gspec, 1), _ in 1:size(eee, gspec, 2), _ in 1:size(eee, gspec, 3)], 
        [zeros(T, N[2]) for _ in 1:size(eeo, gspec, 1), _ in 1:size(eeo, gspec, 2), _ in 1:size(eeo, gspec, 3)],
        [zeros(T, N[3]) for _ in 1:size(eoe, gspec, 1), _ in 1:size(eoe, gspec, 2), _ in 1:size(eoe, gspec, 3)],
        [zeros(T, N[4]) for _ in 1:size(eoo, gspec, 1), _ in 1:size(eoo, gspec, 2), _ in 1:size(eoo, gspec, 3)],
        [zeros(T, N[5]) for _ in 1:size(oee, gspec, 1), _ in 1:size(oee, gspec, 2), _ in 1:size(oee, gspec, 3)],
        [zeros(T, N[6]) for _ in 1:size(oeo, gspec, 1), _ in 1:size(oeo, gspec, 2), _ in 1:size(oeo, gspec, 3)],
        [zeros(T, N[7]) for _ in 1:size(ooe, gspec, 1), _ in 1:size(ooe, gspec, 2), _ in 1:size(ooe, gspec, 3)],
        [zeros(T, N[8]) for _ in 1:size(ooo, gspec, 1), _ in 1:size(ooo, gspec, 2), _ in 1:size(ooo, gspec, 3)])
end

function ones(::Type{VStaggeredGridVariable{T}}, N::Vector{Int}, gspec::GridSpec) where T
    eee, eeo, eoe, eoo, oee, oeo, ooe, ooo = all_eos()
    return VStaggeredGridVariable{T}(
        [ones(T, N[1]) for _ in 1:size(eee, gspec, 1), _ in 1:size(eee, gspec, 2), _ in 1:size(eee, gspec, 3)], 
        [ones(T, N[2]) for _ in 1:size(eeo, gspec, 1), _ in 1:size(eeo, gspec, 2), _ in 1:size(eeo, gspec, 3)],
        [ones(T, N[3]) for _ in 1:size(eoe, gspec, 1), _ in 1:size(eoe, gspec, 2), _ in 1:size(eoe, gspec, 3)],
        [ones(T, N[4]) for _ in 1:size(eoo, gspec, 1), _ in 1:size(eoo, gspec, 2), _ in 1:size(eoo, gspec, 3)],
        [ones(T, N[5]) for _ in 1:size(oee, gspec, 1), _ in 1:size(oee, gspec, 2), _ in 1:size(oee, gspec, 3)],
        [ones(T, N[6]) for _ in 1:size(oeo, gspec, 1), _ in 1:size(oeo, gspec, 2), _ in 1:size(oeo, gspec, 3)],
        [ones(T, N[7]) for _ in 1:size(ooe, gspec, 1), _ in 1:size(ooe, gspec, 2), _ in 1:size(ooe, gspec, 3)],
        [ones(T, N[8]) for _ in 1:size(ooo, gspec, 1), _ in 1:size(ooo, gspec, 2), _ in 1:size(ooo, gspec, 3)])
end

zeros(::Type{VStaggeredGridVariable{T}}, N::Int, gspec::GridSpec) where T = zeros(VStaggeredGridVariable{T}, [N, N, N, N, N, N, N, N], gspec)
ones(::Type{VStaggeredGridVariable{T}}, N::Int, gspec::GridSpec) where T = ones(VStaggeredGridVariable{T}, [N, N, N, N, N, N, N, N], gspec)


const MStaggeredGridVariable{NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T} = StaggeredGridVariable{MVector{NEEE, T}, MVector{NEEO, T}, MVector{NEOE, T}, MVector{NEOO, T}, MVector{NOEE, T}, MVector{NOEO, T}, MVector{NOOE, T}, MVector{NOOO, T}}

function zeros(::Type{MStaggeredGridVariable{NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T}}, gspec::GridSpec) where {NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T}
    eee, eeo, eoe, eoo, oee, oeo, ooe, ooo = all_eos()
    return  StaggeredGridVariable{MVector{NEEE, T}, MVector{NEEO, T}, MVector{NEOE, T}, MVector{NEOO, T}, MVector{NOEE, T}, MVector{NOEO, T}, MVector{NOOE, T}, MVector{NOOO, T}}(
        [zeros(MVector{NEEE, T}) for _ in 1:size(eee, gspec, 1), _ in 1:size(eee, gspec, 2), _ in 1:size(eee, gspec, 3)], 
        [zeros(MVector{NEEO, T}) for _ in 1:size(eeo, gspec, 1), _ in 1:size(eeo, gspec, 2), _ in 1:size(eeo, gspec, 3)],
        [zeros(MVector{NEOE, T}) for _ in 1:size(eoe, gspec, 1), _ in 1:size(eoe, gspec, 2), _ in 1:size(eoe, gspec, 3)],
        [zeros(MVector{NEOO, T}) for _ in 1:size(eoo, gspec, 1), _ in 1:size(eoo, gspec, 2), _ in 1:size(eoo, gspec, 3)],
        [zeros(MVector{NOEE, T}) for _ in 1:size(oee, gspec, 1), _ in 1:size(oee, gspec, 2), _ in 1:size(oee, gspec, 3)],
        [zeros(MVector{NOEO, T}) for _ in 1:size(oeo, gspec, 1), _ in 1:size(oeo, gspec, 2), _ in 1:size(oeo, gspec, 3)],
        [zeros(MVector{NOOE, T}) for _ in 1:size(ooe, gspec, 1), _ in 1:size(ooe, gspec, 2), _ in 1:size(ooe, gspec, 3)],
        [zeros(MVector{NOOO, T}) for _ in 1:size(ooo, gspec, 1), _ in 1:size(ooo, gspec, 2), _ in 1:size(ooo, gspec, 3)])
end

function ones(::Type{MStaggeredGridVariable{NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T}}, gspec::GridSpec) where {NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T}
    eee, eeo, eoe, eoo, oee, oeo, ooe, ooo = all_eos()
    return  StaggeredGridVariable{MVector{NEEE, T}, MVector{NEEO, T}, MVector{NEOE, T}, MVector{NEOO, T}, MVector{NOEE, T}, MVector{NOEO, T}, MVector{NOOE, T}, MVector{NOOO, T}}(
        [ones(MVector{NEEE, T}) for _ in 1:size(eee, gspec, 1), _ in 1:size(eee, gspec, 2), _ in 1:size(eee, gspec, 3)], 
        [ones(MVector{NEEO, T}) for _ in 1:size(eeo, gspec, 1), _ in 1:size(eeo, gspec, 2), _ in 1:size(eeo, gspec, 3)],
        [ones(MVector{NEOE, T}) for _ in 1:size(eoe, gspec, 1), _ in 1:size(eoe, gspec, 2), _ in 1:size(eoe, gspec, 3)],
        [ones(MVector{NEOO, T}) for _ in 1:size(eoo, gspec, 1), _ in 1:size(eoo, gspec, 2), _ in 1:size(eoo, gspec, 3)],
        [ones(MVector{NOEE, T}) for _ in 1:size(oee, gspec, 1), _ in 1:size(oee, gspec, 2), _ in 1:size(oee, gspec, 3)],
        [ones(MVector{NOEO, T}) for _ in 1:size(oeo, gspec, 1), _ in 1:size(oeo, gspec, 2), _ in 1:size(oeo, gspec, 3)],
        [ones(MVector{NOOE, T}) for _ in 1:size(ooe, gspec, 1), _ in 1:size(ooe, gspec, 2), _ in 1:size(ooe, gspec, 3)],
        [ones(MVector{NOOO, T}) for _ in 1:size(ooo, gspec, 1), _ in 1:size(ooo, gspec, 2), _ in 1:size(ooo, gspec, 3)])
end

const SStaggeredGridVariable{NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T} = StaggeredGridVariable{SVector{NEEE, T}, SVector{NEEO, T}, SVector{NEOE, T}, SVector{NEOO, T}, SVector{NOEE, T}, SVector{NOEO, T}, SVector{NOOE, T}, SVector{NOOO, T}}

function zeros(::Type{SStaggeredGridVariable{NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T}}, gspec::GridSpec) where {NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T}
    eee, eeo, eoe, eoo, oee, oeo, ooe, ooo = all_eos()
    return  StaggeredGridVariable{SVector{NEEE, T}, SVector{NEEO, T}, SVector{NEOE, T}, SVector{NEOO, T}, SVector{NOEE, T}, SVector{NOEO, T}, SVector{NOOE, T}, SVector{NOOO, T}}(
        [zeros(SVector{NEEE, T}) for _ in 1:size(eee, gspec, 1), _ in 1:size(eee, gspec, 2), _ in 1:size(eee, gspec, 3)], 
        [zeros(SVector{NEEO, T}) for _ in 1:size(eeo, gspec, 1), _ in 1:size(eeo, gspec, 2), _ in 1:size(eeo, gspec, 3)],
        [zeros(SVector{NEOE, T}) for _ in 1:size(eoe, gspec, 1), _ in 1:size(eoe, gspec, 2), _ in 1:size(eoe, gspec, 3)],
        [zeros(SVector{NEOO, T}) for _ in 1:size(eoo, gspec, 1), _ in 1:size(eoo, gspec, 2), _ in 1:size(eoo, gspec, 3)],
        [zeros(SVector{NOEE, T}) for _ in 1:size(oee, gspec, 1), _ in 1:size(oee, gspec, 2), _ in 1:size(oee, gspec, 3)],
        [zeros(SVector{NOEO, T}) for _ in 1:size(oeo, gspec, 1), _ in 1:size(oeo, gspec, 2), _ in 1:size(oeo, gspec, 3)],
        [zeros(SVector{NOOE, T}) for _ in 1:size(ooe, gspec, 1), _ in 1:size(ooe, gspec, 2), _ in 1:size(ooe, gspec, 3)],
        [zeros(SVector{NOOO, T}) for _ in 1:size(ooo, gspec, 1), _ in 1:size(ooo, gspec, 2), _ in 1:size(ooo, gspec, 3)])
end

function ones(::Type{SStaggeredGridVariable{NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T}}, gspec::GridSpec) where {NEEE, NEEO, NEOE, NEOO, NOEE, NOEO, NOOE, NOOO, T}
    eee, eeo, eoe, eoo, oee, oeo, ooe, ooo = all_eos()
    return  StaggeredGridVariable{SVector{NEEE, T}, SVector{NEEO, T}, SVector{NEOE, T}, SVector{NEOO, T}, SVector{NOEE, T}, SVector{NOEO, T}, SVector{NOOE, T}, SVector{NOOO, T}}(
        [ones(SVector{NEEE, T}) for _ in 1:size(eee, gspec, 1), _ in 1:size(eee, gspec, 2), _ in 1:size(eee, gspec, 3)], 
        [ones(SVector{NEEO, T}) for _ in 1:size(eeo, gspec, 1), _ in 1:size(eeo, gspec, 2), _ in 1:size(eeo, gspec, 3)],
        [ones(SVector{NEOE, T}) for _ in 1:size(eoe, gspec, 1), _ in 1:size(eoe, gspec, 2), _ in 1:size(eoe, gspec, 3)],
        [ones(SVector{NEOO, T}) for _ in 1:size(eoo, gspec, 1), _ in 1:size(eoo, gspec, 2), _ in 1:size(eoo, gspec, 3)],
        [ones(SVector{NOEE, T}) for _ in 1:size(oee, gspec, 1), _ in 1:size(oee, gspec, 2), _ in 1:size(oee, gspec, 3)],
        [ones(SVector{NOEO, T}) for _ in 1:size(oeo, gspec, 1), _ in 1:size(oeo, gspec, 2), _ in 1:size(oeo, gspec, 3)],
        [ones(SVector{NOOE, T}) for _ in 1:size(ooe, gspec, 1), _ in 1:size(ooe, gspec, 2), _ in 1:size(ooe, gspec, 3)],
        [ones(SVector{NOOO, T}) for _ in 1:size(ooo, gspec, 1), _ in 1:size(ooo, gspec, 2), _ in 1:size(ooo, gspec, 3)])
end

const ZerosStaggeredGridVariable{T} = StaggeredGridVariable{Zeros{T, 1, Tuple{Base.OneTo{Int64}}}, Zeros{T, 1, Tuple{Base.OneTo{Int64}}}, Zeros{T, 1, Tuple{Base.OneTo{Int64}}}, Zeros{T, 1, Tuple{Base.OneTo{Int64}}}, Zeros{T, 1, Tuple{Base.OneTo{Int64}}}, Zeros{T, 1, Tuple{Base.OneTo{Int64}}}, Zeros{T, 1, Tuple{Base.OneTo{Int64}}}, Zeros{T, 1, Tuple{Base.OneTo{Int64}}}}

function zeros(::Type{ZerosStaggeredGridVariable{T}}, N::Vector{Int}, gspec::GridSpec) where {T}
    eee, eeo, eoe, eoo, oee, oeo, ooe, ooo = all_eos()
    return  StaggeredGridVariable{Zeros{T, 1, Tuple{Base.OneTo{Int64}}}, Zeros{T, 1, Tuple{Base.OneTo{Int64}}}, Zeros{T, 1, Tuple{Base.OneTo{Int64}}}, Zeros{T, 1, Tuple{Base.OneTo{Int64}}}, Zeros{T, 1, Tuple{Base.OneTo{Int64}}}, Zeros{T, 1, Tuple{Base.OneTo{Int64}}}, Zeros{T, 1, Tuple{Base.OneTo{Int64}}}, Zeros{T, 1, Tuple{Base.OneTo{Int64}}}}(
        [Zeros(zeros(T, N[1])) for _ in 1:size(eee, gspec, 1), _ in 1:size(eee, gspec, 2), _ in 1:size(eee, gspec, 3)], 
        [Zeros(zeros(T, N[2])) for _ in 1:size(eeo, gspec, 1), _ in 1:size(eeo, gspec, 2), _ in 1:size(eeo, gspec, 3)],
        [Zeros(zeros(T, N[3])) for _ in 1:size(eoe, gspec, 1), _ in 1:size(eoe, gspec, 2), _ in 1:size(eoe, gspec, 3)],
        [Zeros(zeros(T, N[4])) for _ in 1:size(eoo, gspec, 1), _ in 1:size(eoo, gspec, 2), _ in 1:size(eoo, gspec, 3)],
        [Zeros(zeros(T, N[5])) for _ in 1:size(oee, gspec, 1), _ in 1:size(oee, gspec, 2), _ in 1:size(oee, gspec, 3)],
        [Zeros(zeros(T, N[6])) for _ in 1:size(oeo, gspec, 1), _ in 1:size(oeo, gspec, 2), _ in 1:size(oeo, gspec, 3)],
        [Zeros(zeros(T, N[7])) for _ in 1:size(ooe, gspec, 1), _ in 1:size(ooe, gspec, 2), _ in 1:size(ooe, gspec, 3)],
        [Zeros(zeros(T, N[8])) for _ in 1:size(ooo, gspec, 1), _ in 1:size(ooo, gspec, 2), _ in 1:size(ooo, gspec, 3)])
end

const BoundaryGridVariable = EvenOddProperty
const BoundaryGridVariableX{OEE, OEO, OOE, OOO} = BoundaryGridVariable{Nothing, Nothing, Nothing, Nothing, Array{OEE, 3}, Array{OEO, 3}, Array{OOE, 3}, Array{OOO, 3}}
const BoundaryGridVariableY{EOE, EOO, OOE, OOO} = BoundaryGridVariable{Nothing, Nothing, Array{EOE, 3}, Array{EOO, 3}, Nothing, Nothing, Array{OOE, 3}, Array{OOO, 3}}
const BoundaryGridVariableZ{EEO, EOO, OEO, OOO} = BoundaryGridVariable{Nothing, Array{EEO, 3}, Nothing, Array{EOO, 3}, Nothing, Array{OEO, 3}, Nothing, Array{OOO, 3}}

const VBoundaryGridVariableX{T} = BoundaryGridVariableX{Vector{T}, Vector{T}, Vector{T}, Vector{T}}
const VBoundaryGridVariableY{T} = BoundaryGridVariableY{Vector{T}, Vector{T}, Vector{T}, Vector{T}}
const VBoundaryGridVariableZ{T} = BoundaryGridVariableZ{Vector{T}, Vector{T}, Vector{T}, Vector{T}}

function zeros(::Type{VBoundaryGridVariableX{T}}, N::Vector{Int}, gspec::GridSpec) where T
    eee, eeo, eoe, eoo, oee, oeo, ooe, ooo = all_eos()
    return VBoundaryGridVariableX{T}(
        nothing, nothing, nothing,  nothing,
        [zeros(T, N[1]) for _ in 1:2, _ in 1:size(oee, gspec, 2), _ in 1:size(oee, gspec, 3)],
        [zeros(T, N[2]) for _ in 1:2, _ in 1:size(oeo, gspec, 2), _ in 1:size(oeo, gspec, 3)],
        [zeros(T, N[3]) for _ in 1:2, _ in 1:size(ooe, gspec, 2), _ in 1:size(ooe, gspec, 3)],
        [zeros(T, N[4]) for _ in 1:2, _ in 1:size(ooo, gspec, 2), _ in 1:size(ooo, gspec, 3)]
    )
end

const MBoundaryGridVariableX{NOEE, NOEO, NOOE, NOOO, T} = BoundaryGridVariableX{MVector{NOEE, T}, MVector{NOEO, T}, MVector{NOOE, T}, MVector{NOOO, T}}
const MBoundaryGridVariableY{NEOE, NEOO, NOOE, NOOO, T} = BoundaryGridVariableY{MVector{NEOE, T}, MVector{NEOO, T}, MVector{NOOE, T}, MVector{NOOO, T}}
const MBoundaryGridVariableZ{NEEO, NEOO, NOEO, NOOO, T} = BoundaryGridVariableZ{MVector{NEEO, T}, MVector{NEOO, T}, MVector{NOEO, T}, MVector{NOOO, T}}

function zeros(::Type{MBoundaryGridVariableX{NOEE, NOEO, NOOE, NOOO, T}}, gspec::GridSpec) where {NOEE, NOEO, NOOE, NOOO, T}
    eee, eeo, eoe, eoo, oee, oeo, ooe, ooo = all_eos()
    return BoundaryGridVariableX{MVector{NOEE, T}, MVector{NOEO, T}, MVector{NOOE, T}, MVector{NOOO, T}}(
        nothing, nothing, nothing,  nothing,
        [zeros(MVector{NOEE, T}) for _ in 1:2, _ in 1:size(oee, gspec, 2), _ in 1:size(oee, gspec, 3)],
        [zeros(MVector{NOEO, T}) for _ in 1:2, _ in 1:size(oeo, gspec, 2), _ in 1:size(oeo, gspec, 3)],
        [zeros(MVector{NOOE, T}) for _ in 1:2, _ in 1:size(ooe, gspec, 2), _ in 1:size(ooe, gspec, 3)],
        [zeros(MVector{NOOO, T}) for _ in 1:2, _ in 1:size(ooo, gspec, 2), _ in 1:size(ooo, gspec, 3)]
    )
end