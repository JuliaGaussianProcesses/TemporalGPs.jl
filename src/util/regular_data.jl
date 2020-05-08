"""
    RegularSpacing{T<:Real} <: AbstractVector{T}

Equivalent to `range(t0; step=Δt, length=N)`, but possible to differentiate through. This
will be removed once it's possible to differentiate through `range`s using `Zygote`.
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
