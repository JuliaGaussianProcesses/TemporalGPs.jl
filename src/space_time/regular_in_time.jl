"""
    RegularInTime{
        T, Tts<:AbstactVector{<:Real}, Tvs<:AbstractVector{<:AbstractVector{T}},
    } <: AbstractVector{T}

Represents data that has multiple observations at each of a given collection of time slices.
"""
struct RegularInTime{
    T, Tts<:AbstractVector{<:Real}, Tvs<:AbstractVector{<:AbstractVector{T}},
} <: AbstractVector{T}
    ts::Tts
    vs::Tvs
end

Base.size(x::RegularInTime) = (sum(length, x.vs), )

function Base.collect(x::RegularInTime)
    time_inputs = vcat([fill(t, length(x)) for (t, x) in zip(x.ts, x.vs)]...)
    space_inputs = vcat(x.vs...)
    return [(x, t) for (x, t) in zip(space_inputs, time_inputs)]
end

Base.show(io::IO, x::RegularInTime) = Base.show(io::IO, collect(x))

get_time(x::RegularInTime) = x.ts

get_space(x::RegularInTime) = x.vs





#
# Implement internal API for transforming between "flat" representation, which is useful for
# GPs, an a time-centric representation, which is useful for state-space models.
#

# See docstring elsewhere for context.
times_from_inputs(x::RegularInTime{<:Real}) = get_time(x)

# See docstring elsewhere for context.
inputs_to_time_form(x::RegularInTime{<:Real}) = get_space(x)

# See docstring elsewhere for context.
function observations_to_time_form(x::RegularInTime, y::AbstractVector{<:Real})
    return restructure(y, length.(get_space(x)))
end

# See docstring elsewhere for context.
function noise_var_to_time_form(x::RegularInTime, S::Diagonal{<:Real})
    vs = restructure(Σ.diag, length.(get_space(x)))
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

function Zygote._pullback(
    ::AContext, ::typeof(restructure), y::Vector, lengths::AbstractVector{<:Integer},
)
    restructure_pullback(Δ::Vector) = nothing, vcat(Δ...), nothing
    return restructure(y, lengths), restructure_pullback
end

# Implementation specific to Fills for AD's sake.
function restructure(y::Fill{<:Real}, lengths::AbstractVector{<:Integer})
    return map(l -> Fill(y.value, l), Zygote.dropgrad(lengths))
end

function restructure(y::AbstractVector, emissions::StructArray)
    return restructure(y, Zygote.dropgrad(map(dim_out, emissions)))
end





# # Old functionality that is potentially redundant.

# """
#     match_to(y::AbstractVector, x::RegularInTime)

# Convert `y` into a vector length `length(x.ts)`, each element containing the number of
# elements in the corresponding element of `x.vs`.
# """
# function match_to(y::AbstractVector{T}, x::RegularInTime) where {T}
#     y_vec = Vector{Vector{T}}(undef, length(x.ts))
#     pos = 1
#     for t in eachindex(x.ts)
#         Nt = length(x.vs[t])
#         y_vec[t] = y[pos:pos + Nt - 1]
#         pos += Nt
#     end
#     return y_vec
# end
