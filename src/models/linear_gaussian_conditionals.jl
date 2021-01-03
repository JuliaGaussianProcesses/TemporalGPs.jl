using Stheno: Xt_invA_X

abstract type AbstractLinearGaussianConditional end

const AbstractLGC = AbstractLinearGaussianConditional

Base.:(==)(x::AbstractLGC, y::AbstractLGC) = (x.A == y.A) && (x.a == y.a) && (x.Q == y.Q)

predict(x::Gaussian, f::AbstractLGC) = Gaussian(f.A * x.m + f.a, f.A * x.P * f.A' + f.Q)

function conditional_rand(rng::AbstractRNG, f::AbstractLGC, x)
    return (f.A * x + f.a) + cholesky(Symmetric(f.Q)).U' * ε_randn(rng, f.a)
end

ε_randn(rng::AbstractRNG, f::AbstractLGC) = ε_randn(rng, f.a)
ε_randn(rng::AbstractRNG, a::Vector{T}) where {T<:Real} = randn(rng, T, length(a))
ε_randn(rng::AbstractRNG, a::T) where {T<:Real} = randn(rng, T)
ε_randn(rng::AbstractRNG, a::T) where {T<:SVector{<:Any, <:Real}} = randn(rng, T)

# @nograd is specialised to `Context`, rather than the more general `AContext` :(
function Zygote._pullback(::AContext, ::typeof(ε_randn), args...)
    ε_randn_pullback(Δ) = nothing
    return ε_randn(args...), ε_randn_pullback
end

scalar_type(x::AbstractVector{T}) where {T} = T
scalar_type(x::T) where {T<:Real} = T

Zygote.@nograd scalar_type

dim_out(f::AbstractLGC) = size(f.A, 1)

dim_in(f::AbstractLGC) = size(f.A, 2)

"""
    LinearGaussianConditional <: AbstractLGC

a.k.a. LGC.
"""
struct LinearGaussianConditional{TA, Ta, TQ} <: AbstractLGC
    A::TA
    a::Ta
    Q::TQ
end

"""
    LGC = LinearGaussianConditional

An alias for LinearGaussianConditional.
"""
const LGC = LinearGaussianConditional

function posterior_and_lml(x::Gaussian, f::LinearGaussianConditional, y)
    A = f.A
    V = A * x.P
    S = cholesky(Symmetric(V * A' + f.Q))
    B = S.U' \ V
    α = S.U' \ (y - (A * x.m + f.a))

    lml = -(length(y) * convert(scalar_type(y), log(2π)) + logdet(S) + α'α) / 2
    return Gaussian(x.m + B'α, x.P - B'B), lml
end

"""
    ScalarOutputLGC

Alias for a LinearGaussianConditional (LGC) which maps from a vector-value to the reals.

In this type, `a` and `Q` should be `Real`s, rather a vector and matrix, and `A` is an
`Adjoint` of an `AbstractVector`.
"""
const ScalarOutputLGC = LinearGaussianConditional{
    <:Adjoint{<:Any, <:AbstractVector}, <:Real, <:Real,
}

"""
    LargeOutputLGC{TA, Ta, TQ} <: AbstractLGC

A LinearGaussianConditional (LGC) specialised for models in which the dimension of the
outputs are greater than that of the inputs. These specialisations both improve numerical
stability and performance (time and memory), so it's worth using if your model lives in
this regime.
"""
struct LargeOutputLGC{TA, Ta, TQ} <: AbstractLGC
    A::TA
    a::Ta
    Q::TQ
end

function posterior_and_lml(x::Gaussian, f::LargeOutputLGC, y)
    A = f.A
    Q = cholesky(Symmetric(f.Q))
    P = cholesky(Symmetric(x.P))

    # Compute posterior covariance matrix.
    B = P.U * A' / Q.U
    F = cholesky(Symmetric(B * B' + UniformScaling(true)))
    G = F.U' \ P.U
    P_post = G'G

    # Compute posterior mean.
    δ = Q.U' \ (y - (A * x.m + f.a))
    β = B * δ
    m_post = x.m + G' * (F.U' \ β)

    # Compute log marginal likelihood.
    c = convert(scalar_type(y), length(y) * log(2π))
    lml = -(δ'δ - Xt_invA_X(F, β) + c + logdet(F) + logdet(Q)) / 2

    return Gaussian(m_post, P_post), lml
end
