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

Base.size(x::RegularInTime) = sum(length, x.vs)

function Base.collect(x::RegularInTime)
    time_inputs = vcat([fill(t, length(x)) for (t, x) in zip(x.ts, x.vs)]...)
    space_inputs = vcat(x.vs...)
    return [(x, t) for (x, t) in zip(space_inputs, time_inputs)]
end

"""
    match_to(y::AbstractVector, x::RegularInTime)

Convert `y` into a vector length `length(x.ts)`, each element containing the number of
elements in the corresponding element of `x.vs`.
"""
function match_to(y::AbstractVector{T}, x::RegularInTime) where {T}
    y_vec = Vector{Vector{T}}(undef, length(x.ts))
    pos = 1
    for t in eachindex(x.ts)
        Nt = length(x.vs[t])
        y_vec[t] = y[pos:pos + Nt - 1]
        pos += Nt
    end
    return y_vec
end

get_time(x::RegularInTime) = x.ts
