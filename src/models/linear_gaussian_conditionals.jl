using Stheno: Xt_invA_X

"""
    abstract type AbstractLGC end


"""
abstract type AbstractLGC end

Base.:(==)(x::AbstractLGC, y::AbstractLGC) = (x.A == y.A) && (x.a == y.a) && (x.Q == y.Q)

Base.eltype(f::AbstractLGC) = eltype(f.A)

"""
    predict(x::Gaussian, f::AbstractLGC)

Compute the distribution "predicted" by this conditional given a `Gaussian` input `x`:
```julia
    Gaussian(f.A * x.m + f.a, f.A * x.P * f.A' + f.Q)
```
"""
predict(x::Gaussian, f::AbstractLGC) = Gaussian(f.A * x.m + f.a, f.A * x.P * f.A' + f.Q)

function predict_marginals(x::Gaussian, f::AbstractLGC)
    return Gaussian(
        f.A * x.m + f.a,
        Diagonal(Stheno.diag_At_B(f.A', x.P * f.A') + diag(f.Q)),
    )
end

function conditional_rand(rng::AbstractRNG, f::AbstractLGC, x::AbstractVector)
    return conditional_rand(ε_randn(rng, f), f, x)
end

function conditional_rand(ε::AbstractVector, f::AbstractLGC, x::AbstractVector)
    return (f.A * x + f.a) + cholesky(Symmetric(f.Q)).U' * ε
end

ε_randn(rng::AbstractRNG, f::AbstractLGC) = ε_randn(rng, f.a)
ε_randn(rng::AbstractRNG, a::AbstractVector{T}) where {T<:Real} = randn(rng, T, length(a))
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
    SmallOutputLGC{
        TA<:AbstractMatrix, Ta<:AbstractVector, TQ<:AbstractMatrix,
    } <: AbstractLGC

a.k.a. LGC. An `AbstractLGC` designed for problems in which `A` is a matrix, and
`size(A, 1) < size(A, 2)`. It should still work (roughly) for problems in which
`size(A, 1) > size(A, 2)`, but one should expect worse accuracy and performance than a
`LargeOutputLGC` in such circumstances.
"""
struct SmallOutputLGC{
    TA<:AbstractMatrix, Ta<:AbstractVector, TQ<:AbstractMatrix,
} <: AbstractLGC
    A::TA
    a::Ta
    Q::TQ
end

function posterior_and_lml(x::Gaussian, f::SmallOutputLGC, y)
    A = f.A
    V = A * x.P
    S = cholesky(Symmetric(V * A' + f.Q))
    B = S.U' \ V
    α = S.U' \ (y - (A * x.m + f.a))

    lml = -(length(y) * convert(scalar_type(y), log(2π)) + logdet(S) + α'α) / 2
    return Gaussian(x.m + B'α, x.P - B'B), lml
end

# Required for type-stability.
function Zygote._pullback(::NoContext, ::Type{<:SmallOutputLGC}, A, a, Q)
    SmallOutputLGC_pullback(::Nothing) = nothing
    SmallOutputLGC_pullback(Δ) = nothing, Δ.A, Δ.a, Δ.Q
    return SmallOutputLGC(A, a, Q), SmallOutputLGC_pullback
end

# This is good progress. Has the potential to improve performance of spatio-temporal things
function Zygote._pullback(::NoContext, ::typeof(+), A::Matrix{<:Real}, D::Diagonal{<:Real})
    function plus_pullback(Δ)
        println("In this one")
        return nothing, Δ, (diag=diag(Δ),)
    end
    return A + D, plus_pullback
end



"""
    LargeOutputLGC{
        TA<:AbstractMatrix, Ta<:AbstractVector, TQ<:AbstractMatrix,
    } <: AbstractLGC

A SmallOutputLGC (LGC) specialised for models in which the dimension of the
outputs are greater than that of the inputs. These specialisations both improve numerical
stability and performance (time and memory), so it's worth using if your model lives in
this regime.
"""
struct LargeOutputLGC{
    TA<:AbstractMatrix, Ta<:AbstractVector, TQ<:AbstractMatrix,
} <: AbstractLGC
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
    lml = _compute_lml(δ, F, β, c, Q)

    return Gaussian(m_post, P_post), lml
end

# For some compiler-y reason, chopping this up helps.
_compute_lml(δ, F, β, c, Q) = -(δ'δ - Xt_invA_X(F, β) + c + logdet(F) + logdet(Q)) / 2


"""
    ScalarOutputLGC

Alias for a SmallOutputLGC (LGC) which maps from a vector-value to the reals.

In this type, `a` and `Q` should be `Real`s, rather a vector and matrix, and `A` is an
`Adjoint` of an `AbstractVector`.
"""
struct ScalarOutputLGC{
    TA<:Adjoint{<:Any, <:AbstractVector}, Ta<:Real, TQ<:Real,
} <: AbstractLGC
    A::TA
    a::Ta
    Q::TQ
end

function conditional_rand(ε::Real, f::ScalarOutputLGC, x::AbstractVector)
    return (f.A * x + f.a) + sqrt(f.Q) * ε
end

ε_randn(rng::AbstractRNG, f::ScalarOutputLGC) = randn(rng, eltype(f))

function posterior_and_lml(x::Gaussian, f::ScalarOutputLGC, y::T) where {T<:Real}
    A = f.A
    V = A * x.P
    sqrtS = sqrt(V * A' + f.Q)
    B = sqrtS \ V
    α = sqrtS \ (y - (A * x.m + f.a))

    lml = -(convert(T, log(2π)) + 2 * log(sqrtS) + α^2) / 2
    return Gaussian(x.m + B'α, x.P - B'B), lml
end
