"""
    RegularInTime{
        Tt, Tv, Tts<:AbstractVector{Tt}, Tvs<:AbstractVector{<:AbstractVector{Tv}},
    } <: AbstractVector{Tuple{Tt, Tv}}

Represents data that has multiple observations at each of a given collection of time slices.
"""
struct RegularInTime{
    Tt, Tv, Tts<:AbstractVector{Tt}, Tvs<:AbstractVector{<:AbstractVector{Tv}},
} <: AbstractVector{Tuple{Tt, Tv}}
    ts::Tts
    vs::Tvs
end

get_space(x::RegularInTime) = Zygote.literal_getfield(x, Val(:vs))

get_times(x::RegularInTime) = Zygote.literal_getfield(x, Val(:ts))

Base.size(x::RegularInTime) = (sum(length, x.vs), )

function Base.collect(x::RegularInTime)
    time_inputs = reduce(vcat, [fill(t, length(x)) for (t, x) in zip(x.ts, x.vs)])
    space_inputs = reduce(vcat, x.vs)
    return [(x, t) for (x, t) in zip(space_inputs, time_inputs)]
end

Base.getindex(x::RegularInTime, n::Int) = collect(x)[n]

Base.show(io::IO, x::RegularInTime) = Base.show(io::IO, collect(x))





#
# Implement internal API for transforming between "flat" representation, which is useful for
# GPs, an a time-centric representation, which is useful for state-space models.
#

# See docstring elsewhere for context.
inputs_to_time_form(x::RegularInTime{<:Real}) = get_space(x)

# See docstring elsewhere for context.
function observations_to_time_form(x::RegularInTime, y::AbstractVector)
    return restructure(y, length.(get_space(x)))
end

# See docstring elsewhere for context.
function noise_var_to_time_form(x::RegularInTime, S::Diagonal{<:Real})
    vs = restructure(S.diag, length.(get_space(x)))
    return Diagonal.(collect.(vs))
end

destructure(::RegularInTime, y::AbstractVector{<:AbstractVector{<:Real}}) = reduce(vcat, y)

function restructure(y::AbstractVector{<:Real}, lengths::AbstractVector{<:Integer})
    idxs_start = cumsum(vcat(0, lengths)) .+ 1
    idxs_end = idxs_start[1:end-1] .+ lengths .- 1
    return map(n -> y[idxs_start[n]:idxs_end[n]], eachindex(lengths))
end

function restructure(y::AbstractVector{T}, lengths::AbstractVector{<:Integer}) where {T}
    idxs_start = cumsum(vcat(0, lengths)) .+ 1
    idxs_end = idxs_start[1:end-1] .+ lengths .- 1
    return map(eachindex(lengths)) do n
        y_missing = Vector{T}(undef, lengths[n])
        y_missing .= y[idxs_start[n]:idxs_end[n]]
        return y_missing
    end
end

function ChainRulesCore.rrule(
    ::typeof(restructure), y::Vector, lengths::AbstractVector{<:Integer},
)
    restructure_pullback(Δ::Vector) = NoTangent(), reduce(vcat, Δ), NoTangent()
    return restructure(y, lengths), restructure_pullback
end

# Implementation specific to Fills for AD's sake.
function restructure(y::Fill{<:Real}, lengths::AbstractVector{<:Integer})
    return map(l -> Fill(y.value, l), ChainRulesCore.ignore_derivatives(lengths))
end

function restructure(y::AbstractVector, emissions::StructArray)
    return restructure(y, ChainRulesCore.ignore_derivatives(map(dim_out, emissions)))
end
