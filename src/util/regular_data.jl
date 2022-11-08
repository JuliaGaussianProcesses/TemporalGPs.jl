"""
    RegularSpacing{T<:Real} <: AbstractVector{T}

`RegularSpacing(t0, Δt, N)` represents the same thing as `range(t0; step=Δt, length=N)`, but
has a different implementation that makes it possible to differentiate through with the
current version of `Zygote`. This data structure will be entirely removed once it's possible
to work with `StepRangeLen`s in `Zygote`.

Relevant issue: https://github.com/FluxML/Zygote.jl/issues/550
"""
struct RegularSpacing{T<:Real} <: AbstractVector{T}
    t0::T
    Δt::T
    N::Int
end

# Implements the AbstractArray interface.

Base.IndexStyle(::RegularSpacing) = Base.IndexLinear()

Base.size(x::RegularSpacing) = (x.N,)

Base.getindex(x::RegularSpacing, n::Int) = x.t0 + (n - 1) * x.Δt

Base.step(x::RegularSpacing) = x.Δt

function ChainRulesCore.rrule(::Type{TR}, t0::T, Δt::T, N::Int) where {TR<:RegularSpacing, T<:Real}
    function pullback_RegularSpacing(Δ::TΔ) where {TΔ<:NamedTuple}
        return (
            NoTangent(),
            hasfield(TΔ, :t0) ? Δ.t0 : NoTangent(),
            hasfield(TΔ, :Δt) ? Δ.Δt : NoTangent(),
            NoTangent(),
        )
    end
    return RegularSpacing(t0, Δt, N), pullback_RegularSpacing
end
