import Random: rand
import Base: isapprox, copy, deepcopy

"""
    Gaussian{Tm, TP}

A (multivariate) Gaussian with mean vector `m` and variance matrix `P`.
This doesn't currently conform to Distributions.jl standards. Work to make this happen
would be welcomed. 
"""
struct Gaussian{Tm, TP}
    m::Tm
    P::TP
end

dim(x::Gaussian) = length(x.m)

Random.rand(rng::AbstractRNG, x::Gaussian) = vec(rand(rng, x, 1))

function Random.rand(rng::AbstractRNG, x::Gaussian, S::Int)
    return x.m .+ cholesky(Symmetric(x.P)).U' * randn(rng, length(x.m), S)
end

Stheno.logpdf(x::Gaussian, y::AbstractVector{<:Real}) = first(logpdf(x, reshape(y, :, 1)))

function Stheno.logpdf(x::Gaussian, Y::AbstractMatrix{<:Real})
    μ, C = x.m, cholesky(Symmetric(x.P))
    T = promote_type(eltype(μ), eltype(C), eltype(Y))
    return -((size(Y, 1) * T(log(2π)) + logdet(C)) .+ Stheno.diag_Xt_invA_X(C, Y .- μ)) ./ 2
end

Base.:(==)(x::Gaussian, y::Gaussian) = x.m == y.m && x.P == y.P

Base.copy(x::Gaussian) = Gaussian(copy(x.m), copy(x.P))
