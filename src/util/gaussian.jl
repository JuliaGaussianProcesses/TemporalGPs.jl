import Random: rand
import Base: isapprox, copy, deepcopy

"""
    Gaussian{Tm, TP}

A (multivariate) Gaussian with mean vector `m` and variance matrix `P`.
This doesn't currently conform to Distributions.jl standards. Work to make this happen
would be welcomed.

It was necessary to implement this a year or so ago for AD-related reasons. It's quite
possible that in the intervening period of time things have improved and this type is no
longer necessary in addition to the `MvNormal` type in `Distributions`. I've not had the
time to remove it though.
"""
struct Gaussian{Tm, TP}
    m::Tm
    P::TP
end

dim(x::Gaussian) = length(x.m)

function get_fields(x::Gaussian)
    m = Zygote.literal_getfield(x, Val(:m))
    P = Zygote.literal_getfield(x, Val(:P))
    return m, P
end

Random.rand(rng::AbstractRNG, x::Gaussian) = vec(rand(rng, x, 1))

Random.rand(rng::AbstractRNG, x::Gaussian{<:SVector}) = randn(rng, typeof(x.m))

function Random.rand(rng::AbstractRNG, x::Gaussian, S::Int)
    P = x.P + UniformScaling(1e-12)
    return x.m .+ cholesky(Symmetric(P)).U' * randn(rng, length(x.m), S)
end

Stheno.logpdf(x::Gaussian, y::AbstractVector{<:Real}) = first(logpdf(x, reshape(y, :, 1)))

function Stheno.logpdf(x::Gaussian, Y::AbstractMatrix{<:Real})
    μ, C = x.m, cholesky(Symmetric(x.P))
    T = promote_type(eltype(μ), eltype(C), eltype(Y))
    return -((size(Y, 1) * T(log(2π)) + logdet(C)) .+ Stheno.diag_Xt_invA_X(C, Y .- μ)) ./ 2
end

Base.:(==)(x::Gaussian, y::Gaussian) = x.m == y.m && x.P == y.P

function Base.isapprox(x::Gaussian, y::Gaussian; kwargs...)
    return isapprox(x.m, y.m; kwargs...) && isapprox(x.P, y.P; kwargs...)
end

Stheno.mean(x::Gaussian) = x.m

Stheno.cov(x::Gaussian) = x.P

Stheno.marginals(x::Gaussian{<:Real, <:Real}) = Normal(x.m, sqrt(x.P))

function Stheno.marginals(x::Gaussian{<:AbstractVector, <:AbstractMatrix})
    return Normal.(x.m, sqrt.(diag(x.P)))
end

storage_type(x::Gaussian{<:SVector{D, T}}) where {D, T<:Real} = SArrayStorage(T)

storage_type(gmm::Gaussian{<:Vector{T}}) where {T<:Real} = ArrayStorage(T)

storage_type(x::Gaussian{T}) where {T<:Real} = ScalarStorage(T)

function Zygote._pullback(::AContext, ::Type{<:Gaussian}, m, P)
    Gaussian_pullback(Δ::Nothing) = (nothing, nothing, nothing)
    Gaussian_pullback(Δ) = (nothing, Δ.m, Δ.P)
    return Gaussian(m, P), Gaussian_pullback
end

Base.length(x::Gaussian) = 0

# Zero-adjoint initialisation for the benefit of `scan`.
_get_zero_adjoint(x::Gaussian) = (m=_get_zero_adjoint(x.m), P=_get_zero_adjoint(x.P))
