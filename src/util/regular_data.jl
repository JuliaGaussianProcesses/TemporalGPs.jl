"""
    RegularSpacing{T<:Real} <: AbstractVector{T}

`RegularSpacing(t0, Δt, N)` represents the same thing as `range(t0; step=Δt, length=N)`, but
has a different implementation that makes it possible to differentiate through with the
current version of `Zygote`. This data structure will be entirely removed once it's possible
to work with `StepRangeLen`s in `Zygote`.

Relevant issue: https://github.com/FluxML/Zygote.jl/issues/550

```jldoctest
julia> x = range(0.0; step=0.2, length=10);

julia> y = RegularSpacing(0.0, 0.2, 10);

julia> x ≈ y
true
```
"""
struct RegularSpacing{T<:Real} <: AbstractVector{T}
    t0::T
    Δt::T
    N::Int
end

# Implement the AbstractArray interface.

Base.IndexStyle(::RegularSpacing) = Base.IndexLinear()

Base.size(x::RegularSpacing) = (x.N,)

Base.getindex(x::RegularSpacing, n::Int) = x.t0 + (n - 1) * x.Δt

Base.step(x::RegularSpacing) = x.Δt

ZygoteRules.@adjoint function (::Type{TR})(t0::T, Δt::T, N::Int) where {TR<:RegularSpacing, T<:Real}
    function pullback_RegularSpacing(Δ::TΔ) where {TΔ<:NamedTuple}
        return (
            hasfield(TΔ, :t0) ? Δ.t0 : nothing,
            hasfield(TΔ, :Δt) ? Δ.Δt : nothing,
            nothing,
        )
    end
    return RegularSpacing(t0, Δt, N), pullback_RegularSpacing
end


"""
    ExtendedRegularSpacing{T<:Real} <: AbstractVector{T}

Add additional points to a `RegularSpacing` in a manner that both contains the original data
points and is also regular.

# Fields:
- `x::RegularSpacing{T}`: original data
- `lhs_extension::Int`: number of additional points to the left (new spacing)
- `rhs_extension::Int`: number of additional points to the right (new spacing)
- `ρ::Int=1`: number of points per interval in the original spacing.

# Examples:

`ExtendedRegularSpacing`s can recover `RegularSpacing`s:
```jldoctest
julia> x = RegularSpacing(0.0, 0.1, 10);

julia> y = ExtendedRegularSpacing(x, 0, 0, 1);

julia> x == y
true

julia> y == ExtendedRegularSpacing(x, 0, 0)
true
```

Extending a `RegularSpacing` in one direction or another simple adds points at either end:
```jldoctest
julia> x = RegularSpacing(0.0, 0.1, 3);

julia> ExtendedRegularSpacing(x, 2, 3) ≈ vcat([-0.2, -0.1], x, [0.3, 0.4, 0.5])
true
```

Increasing the density adds points between existing points:
```jldoctest
julia> x = RegularSpacing(0.0, 0.1, 3);

julia> ExtendedRegularSpacing(x, 0, 0, 2) ≈ [0.0, 0.05, 0.1, 0.15, 0.2]
true
```

Doing both extends at the new density:
```jldoctest
julia> x = RegularSpacing(0.0, 0.1, 3);

julia> ExtendedRegularSpacing(x, 1, 0, 2) ≈ [-0.05, 0.0, 0.05, 0.1, 0.15, 0.2]
true
```
"""
struct ExtendedRegularSpacing{T<:Real, Tx<:RegularSpacing{T}} <: AbstractVector{T}
    x::Tx
    lhs_extension::Int
    rhs_extension::Int
    ρ::Int
end

function ExtendedRegularSpacing(x::RegularSpacing, lhs_extension::Int, rhs_extension::Int)
    return ExtendedRegularSpacing(x, lhs_extension, rhs_extension, 1)
end

# Implement the AbstractArray interface.

Base.IndexStyle(::ExtendedRegularSpacing) = Base.IndexLinear()

function Base.size(x::ExtendedRegularSpacing)
    return ((length(x.x) - 1) * x.ρ + 1 + x.lhs_extension + x.rhs_extension,)
end

function Base.getindex(x::ExtendedRegularSpacing, n::Int)
    return (x.x[1] - step(x) * x.lhs_extension) + (n - 1) * step(x)
end

Base.step(x::ExtendedRegularSpacing) = step(x.x) / x.ρ

function Base.convert(::Type{<:RegularSpacing}, x::ExtendedRegularSpacing)
    return RegularSpacing(first(x), step(x), length(x))
end
