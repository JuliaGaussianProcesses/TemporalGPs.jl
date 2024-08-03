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

AbstractGPs.mean(x::Gaussian) = x.m

AbstractGPs.cov(x::Gaussian) = x.P

AbstractGPs.var(x::Gaussian{<:AbstractVector}) = diag(cov(x))

AbstractGPs.var(x::Gaussian{<:Real}) = cov(x)

get_fields(x::Gaussian) = mean(x), cov(x)

Random.rand(rng::AbstractRNG, x::Gaussian) = vec(rand(rng, x, 1))

function Random.rand(rng::AbstractRNG, x::Gaussian{Tm}) where {Tm<:SVector}
    ε = randn(rng, Tm)
    return mean(x) + cholesky(Symmetric(cov(x) + UniformScaling(1e-12))).U' * ε
end

function Random.rand(rng::AbstractRNG, x::Gaussian, S::Int)
    P = cov(x) + UniformScaling(1e-12)
    return mean(x) .+ cholesky(Symmetric(P)).U' * randn(rng, length(mean(x)), S)
end

function AbstractGPs.logpdf(x::Gaussian, y::AbstractVector{<:Real})
    return first(logpdf(x, reshape(y, :, 1)))
end

function AbstractGPs.logpdf(x::Gaussian, Y::AbstractMatrix{<:Real})
    μ, C = mean(x), cholesky(Symmetric(cov(x)))
    T = promote_type(eltype(μ), eltype(C), eltype(Y))
    return -((size(Y, 1) * T(log(2π)) + logdet(C)) .+ diag_Xt_invA_X(C, Y .- μ)) ./ 2
end

Base.:(==)(x::Gaussian, y::Gaussian) = mean(x) == mean(y) && cov(x) == cov(y)

function Base.isapprox(x::Gaussian, y::Gaussian; kwargs...)
    return isapprox(mean(x), mean(y); kwargs...) && isapprox(cov(x), cov(y); kwargs...)
end

function AbstractGPs.marginals(x::Gaussian{T, T}) where {T<:Real}
    return AbstractGPs.Normal{T}(mean(x), sqrt(cov(x)))
end

function AbstractGPs.marginals(x::Gaussian{<:AbstractVector, <:AbstractMatrix})
    return AbstractGPs.Normal.(mean(x), sqrt.(diag(cov(x))))
end

storage_type(::Gaussian{<:Vector{T}}) where {T<:Real} = ArrayStorage(T)
storage_type(::Gaussian{<:SVector{D, T}}) where {D, T<:Real} = SArrayStorage(T)
storage_type(::Gaussian{T}) where {T<:Real} = ScalarStorage(T)

function ChainRulesCore.rrule(::Type{<:Gaussian}, m, P)
    proj_P = ProjectTo(P)
    Gaussian_pullback(::ZeroTangent) = NoTangent(), NoTangent(), NoTangent()
    Gaussian_pullback(Δ) = NoTangent(), Δ.m, proj_P(Δ.P)
    return Gaussian(m, P), Gaussian_pullback
end

Base.length(x::Gaussian) = 0

# Zero-adjoint initialisation for the benefit of `scan`.
_get_zero_adjoint(x::Gaussian) = (m=_get_zero_adjoint(mean(x)), P=_get_zero_adjoint(cov(x)))
