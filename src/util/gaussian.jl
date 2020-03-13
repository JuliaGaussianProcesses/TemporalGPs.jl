import Random: rand
import Base: isapprox, ==, copy, deepcopy

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

function Random.rand(rng::AbstractRNG, x::Gaussian)
    return x.m + cholesky(x.P).U' * randn(rng, length(x.m))
end

function Random.rand(rng::AbstractRNG, x::Gaussian, S::Int)
    return x.m .+ cholesky(Symmetric(x.P)).U' * randn(rng, length(x.m), S)
end
deepcopy(x::Gaussian) = Gaussian(deepcopy(x.m), deepcopy(x.P))
copy(x::Gaussian) = Gaussian(copy(x.m), copy(x.P))

==(x::Gaussian, y::Gaussian) = x.m == y.m && x.P == y.P

function isapprox(x::Gaussian{<:AV, <:AM}, y::Gaussian{<:AV, <:AM})
    return isapprox(x.m, y.m) && isapprox(x.P, y.P)
end

function isapprox(x::Gaussian{<:AV, <:AM}, y::Gaussian{<:Real, <:Real})
    return length(x.m) == 1 && isapprox(first(x.m), y.m) && isapprox(first(x.P), y.P)
end

isapprox(y::Gaussian{<:Real, <:Real}, x::Gaussian{<:AV, <:AM}) = isapprox(x, y)
