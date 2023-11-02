## define single even odd
abstract type EvenOdd end
struct Even <: EvenOdd end
struct Odd <: EvenOdd end

switch_eo(::Type{Even}) = Odd
switch_eo(::Type{Odd}) = Even

## define the 3D even odd classification
struct EvenOddClassification{EOX <: EvenOdd, EOY <: EvenOdd, EOZ <: EvenOdd} end

eo_in(::Type{EvenOddClassification{EOX, EOY, EOZ}}, ::Type{X}) where {EOX <: EvenOdd, EOY <: EvenOdd, EOZ <: EvenOdd} = EOX
eo_in(::Type{EvenOddClassification{EOX, EOY, EOZ}}, ::Type{Y}) where {EOX <: EvenOdd, EOY <: EvenOdd, EOZ <: EvenOdd} = EOY
eo_in(::Type{EvenOddClassification{EOX, EOY, EOZ}}, ::Type{Z}) where {EOX <: EvenOdd, EOY <: EvenOdd, EOZ <: EvenOdd} = EOZ

is_even_in(::Type{EvenOddClassification{EOX, EOY, EOZ}}, ::Type{D}) where {EOX <: EvenOdd, EOY <: EvenOdd, EOZ <: EvenOdd, D <: Dimension} = eo_in(EvenOddClassification{EOX, EOY, EOZ}, D) == Even
is_odd_in(::Type{EvenOddClassification{EOX, EOY, EOZ}}, ::Type{D}) where {EOX <: EvenOdd, EOY <: EvenOdd, EOZ <: EvenOdd, D <: Dimension} = eo_in(EvenOddClassification{EOX, EOY, EOZ}, D) == Odd

all_eos() = (EvenOddClassification{EOX, EOY, EOZ} for EOX in (Even, Odd) for EOY in (Even, Odd) for EOZ in (Even, Odd))
all_eos_eo_in(::Type{EO}, ::Type{D}) where {EO <: EvenOdd, D <: Dimension}= (eo for eo in all_eos() if eo_in(eo, D) == EO)
all_eos_odd_in(::Type{D}) where {D <: Dimension} = all_eos_eo_in(Odd, D)
all_eos_even_in(::Type{D}) where {D <: Dimension}= all_eos_eo_in(Even, D)

switch_eo(::Type{EvenOddClassification{EOX, EOY, EOZ}}, ::Type{X}) where {EOX, EOY, EOZ} = EvenOddClassification{switch_eo(EOX), EOY, EOZ}
switch_eo(::Type{EvenOddClassification{EOX, EOY, EOZ}}, ::Type{Y}) where {EOX, EOY, EOZ} = EvenOddClassification{EOX, switch_eo(EOY), EOZ}
switch_eo(::Type{EvenOddClassification{EOX, EOY, EOZ}}, ::Type{Z}) where {EOX, EOY, EOZ} = EvenOddClassification{EOX, EOY, switch_eo(EOZ)}
## define a parametric type, that can be indexed by EvenOddClassification
struct EvenOddProperty{EEE, EEO, EOE, EOO, OEE, OEO, OOE, OOO}
        eee::EEE
        eeo::EEO
        eoe::EOE
        eoo::EOO
        oee::OEE
        oeo::OEO
        ooe::OOE
        ooo::OOO
end

# implement "indexing" by even odd classification / returns the field of the object

to_symbol(::Type{EvenOddClassification{Even, Even, Even}}) = :eee
to_symbol(::Type{EvenOddClassification{Even, Even, Odd}}) = :eeo
to_symbol(::Type{EvenOddClassification{Even, Odd, Even}}) = :eoe
to_symbol(::Type{EvenOddClassification{Even, Odd, Odd}}) = :eoo
to_symbol(::Type{EvenOddClassification{Odd, Even, Even}}) = :oee
to_symbol(::Type{EvenOddClassification{Odd, Even, Odd}}) = :oeo
to_symbol(::Type{EvenOddClassification{Odd, Odd, Even}}) = :ooe
to_symbol(::Type{EvenOddClassification{Odd, Odd, Odd}}) = :ooo

import Base.getindex
getindex(object::EvenOddProperty, eo::Type{<: EvenOddClassification}) = getproperty(object, to_symbol(eo))

import Base: length, iterate
function getindex(object::EvenOddProperty, i::Int)
    @assert 1 <= i <= 8
    return getproperty(object, [:eee, :eeo, :eoe, :eoo, :oee, :oeo, :ooe, :ooo][i])
end

length(object::EvenOddProperty) = 8
iterate(object::EvenOddProperty, i=1) = i > length(object) ? nothing : (object[i], i+1)

## define even odd pairity
abstract type EvenOddPairity end
struct EvenPairity <: EvenOddPairity end
struct OddPairity <: EvenOddPairity end

switch_eop(::Type{EvenPairity}) = OddPairity
switch_eop(::Type{OddPairity}) = EvenPairity

# pairity is even: eee, ooe, oeo, eoo
# pairity is odd: oee, eoe, eeo, ooo

pairity(::Type{EvenOddClassification{Even, Even, Even}}) = EvenPairity
pairity(::Type{EvenOddClassification{Odd, Odd, Even}}) = EvenPairity
pairity(::Type{EvenOddClassification{Odd, Even, Odd}}) = EvenPairity
pairity(::Type{EvenOddClassification{Even, Odd, Odd}}) = EvenPairity

pairity(::Type{EvenOddClassification{Odd, Even, Even}}) = OddPairity
pairity(::Type{EvenOddClassification{Even, Odd, Even}}) = OddPairity
pairity(::Type{EvenOddClassification{Even, Even, Odd}}) = OddPairity
pairity(::Type{EvenOddClassification{Odd, Odd, Odd}}) = OddPairity

is_even_pairity(::Type{EvenOddClassification{EOX, EOY, EOZ}}) where {EOX <: EvenOdd, EOY <: EvenOdd, EOZ <: EvenOdd} = pairity(EvenOddClassification{EOX, EOY, EOZ}) == EvenPairity
is_odd_pairity(::Type{EvenOddClassification{EOX, EOY, EOZ}}) where {EOX <: EvenOdd, EOY <: EvenOdd, EOZ <: EvenOdd} = pairity(EvenOddClassification{EOX, EOY, EOZ}) == OddPairity

all_eos_of_eop(::Type{EOP}) where {EOP <: EvenOddPairity} = (eo for eo in all_eos() if pairity(eo) == EOP)
all_eos_of_even_pairity() = all_eos_of_eop(EvenPairity)
all_eos_of_odd_pairity() = all_eos_of_eop(OddPairity)
