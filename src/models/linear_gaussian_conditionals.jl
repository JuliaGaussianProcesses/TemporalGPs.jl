"""
    AbstractLGC

Represents a Gaussian conditional distribution:

```julia
y | x ∼ Gaussian(A * x + a, Q)
```

Note that this can be used in two contexts:

- Transition: `y` is the next state, `x` is the current state.
- Emission: `y` is the observation, `x` is the state.

Subtypes have discretion over how to implement the interface for this type. In particular
`A`, `a`, and `Q` may not be represented explicitly so that structure can be exploited to
accelerate computations.

## Interface:
- `==`
- `eltype`
- `dim_out`
- `dim_in`
- `predict`
- `predict_marginals`
- `conditional_rand`
- `ε_randn`
- `posterior_and_lml`
"""
abstract type AbstractLGC end

Base.:(==)(x::AbstractLGC, y::AbstractLGC) = (x.A == y.A) && (x.a == y.a) && (x.Q == y.Q)

Base.eltype(f::AbstractLGC) = eltype(f.A)

"""
    predict(x::Gaussian, f::AbstractLGC)::Gaussian{Tm,AbstractMatrix}

Compute the distribution "predicted" by this conditional given a [`Gaussian`](@ref) input `x`. Will
be equivalent to

```julia
    Gaussian(f.A * x.m + f.a, f.A * x.P * f.A' + f.Q)
```
"""
function predict(x::Gaussian, f::AbstractLGC)
    A, a, Q = get_fields(f)
    m, P = get_fields(x)
    
    # Symmetric wrapper needed for numerical stability. Do not unwrap.
    return Gaussian(A * m + a, (A * symmetric(P)) * A' + Q)
end

"""
    predict_marginals(x::Gaussian, f::AbstractLGC)::Gaussian{Tm,Diagonal}

Equivalent to
```julia
    xꜝ⁺¹ = predict(xꜝ, f)
    Gaussian(mean(xꜝ⁺¹), Diagonal(cov(xꜝ⁺¹)))
```
"""
function predict_marginals(x::Gaussian, f::AbstractLGC)
    return Gaussian(
        f.A * x.m + f.a,
        Diagonal(diag_At_B(f.A', x.P * f.A') + diag(f.Q)),
    )
end

"""
    conditional_rand(rng::AbstractRNG, f::AbstractLGC, x::AbstractVector)
    conditional_rand(ε::AbstractVector, f::AbstractLGC, x::AbstractVector)

Sample from the conditional distribution `y | x`. `ε` is the randomness needed to generate
this sample. If `rng` is provided, it will be used to construct `ε` via [`ε_randn`](@ref).

If implementing a new `AbstractLGC`, implement the `ε` method as it avoids randomness, which
means that it plays nicely with `scan_emit`'s checkpointed rrule.
"""
function conditional_rand(rng::AbstractRNG, f::AbstractLGC, x::AbstractVector)
    return conditional_rand(ε_randn(rng, f), f, x)
end

function conditional_rand(ε::AbstractVector, f::AbstractLGC, x::AbstractVector)
    A, a, Q = get_fields(f)
    return (A * x + a) + cholesky(symmetric(Q + UniformScaling(1e-9))).U' * ε
end

"""
    ε_randn(rng::AbstractRNG, f::AbstractLGC)

Generate the vector of random numbers needed inside [`conditional_rand`](@ref).
"""
ε_randn(rng::AbstractRNG, f::AbstractLGC) = ε_randn(rng, f.A)
ε_randn(rng::AbstractRNG, A::AbstractMatrix{T}) where {T<:Real} = randn(rng, T, size(A, 1))
function ε_randn(rng::AbstractRNG, ::SMatrix{Dout, Din, T}) where {Dout, Din, T<:Real}
    return randn(rng, SVector{Dout, T})
end

scalar_type(::AbstractVector{T}) where {T} = T
scalar_type(::T) where {T<:Real} = T

"""
    SmallOutputLGC{
        TA<:AbstractMatrix, Ta<:AbstractVector, TQ<:AbstractMatrix,
    } <: AbstractLGC

a.k.a. LGC. An [`AbstractLGC`](@ref) designed for problems in which `A` is a matrix, and
`size(A, 1) < size(A, 2)`. It should still work (roughly) for problems in which
`size(A, 1) > size(A, 2)`, but one should expect worse accuracy and performance than a
[`LargeOutputLGC`](@ref) in such circumstances.
"""
struct SmallOutputLGC{
    TA<:AbstractMatrix, Ta<:AbstractVector, TQ<:AbstractMatrix,
} <: AbstractLGC
    A::TA
    a::Ta
    Q::TQ
end

dim_out(f::SmallOutputLGC) = size(f.A, 1)

dim_in(f::SmallOutputLGC) = size(f.A, 2)

noise_cov(f::SmallOutputLGC) = f.Q

get_fields(f::SmallOutputLGC) = (f.A, f.a, f.Q)

function posterior_and_lml(x::Gaussian, f::SmallOutputLGC, y::AbstractVector{<:Real})
    m, P = get_fields(x)
    A, a, Q = get_fields(f)

    V = A * P

    S = cholesky(symmetric(V * A' + Q))
    B = S.U' \ V
    α = S.U' \ (y - (A * m + a))

    lml = -(length(y) * convert(scalar_type(y), log(2π)) + logdet(S) + α'α) / 2
    return Gaussian(m + B'α, P - B'B), lml
end

function posterior_and_lml(
    x::Gaussian, f::SmallOutputLGC, y::AbstractVector{<:Union{Missing, <:Real}},
)
    # This implicitly assumes that Q is Diagonal. MethodError if not.
    A, a, Q = get_fields(f)
    Q_filled, y_filled = fill_in_missings(Q, y)
    x_post, lml_raw = posterior_and_lml(x, SmallOutputLGC(A, a, Q_filled), y_filled)
    return x_post, lml_raw + _logpdf_volume_compensation(y)
end

"""
    LargeOutputLGC{
        TA<:AbstractMatrix, Ta<:AbstractVector, TQ<:AbstractMatrix,
    } <: AbstractLGC

A [`SmallOutputLGC`](@ref) (LGC) specialised for models in which the dimension of the
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

dim_out(f::LargeOutputLGC) = size(f.A, 1)

dim_in(f::LargeOutputLGC) = size(f.A, 2)

noise_cov(f::LargeOutputLGC) = f.Q

get_fields(f::LargeOutputLGC) = (f.A, f.a, f.Q)

function posterior_and_lml(x::Gaussian, f::LargeOutputLGC, y::AbstractVector{<:Real})
    m, _P = get_fields(x)
    A, a, _Q = get_fields(f)
    Q = cholesky(symmetric(_Q))
    P = cholesky(symmetric(_P + ident_eps(1e-10)))

    # Compute posterior covariance matrix.
    Bt = Q.U' \ A * P.U'
    F = cholesky(symmetric(Bt' * Bt + UniformScaling(1.0)))
    G = F.U' \ P.U
    P_post = G' * G

    # Compute posterior mean.
    δ = Q.U' \ (y - (A * m + a))
    β = F.U' \ (Bt' * δ)
    m_post = m + G' * β

    # Compute log marginal likelihood.
    c = convert(scalar_type(y), length(y) * log(2π))
    lml = _compute_lml(δ, F, β, c, Q)

    return Gaussian(m_post, P_post), lml
end

# For some compiler-y reason, chopping this up helps.
_compute_lml(δ, F, β, c, Q) = -(δ'δ - β'β + c + logdet(F) + logdet(Q)) / 2

function posterior_and_lml(
    x::Gaussian, f::LargeOutputLGC, y::AbstractVector{<:Union{Missing, <:Real}},
)
    A, a, Q = get_fields(f)
    # This implicitly assumes that Q is Diagonal. MethodError if not.
    Q_filled, y_filled = fill_in_missings(Q, y)
    x_post, lml_raw = posterior_and_lml(x, LargeOutputLGC(A, a, Q_filled), y_filled)
    return x_post, lml_raw + _logpdf_volume_compensation(y)
end



"""
    ScalarOutputLGC

An [`AbstractLGC`](@ref) that operates on a vector-valued input space and a scalar-valued output space.
Similar to [`SmallOutputLGC`](@ref) when its `dim_out` is 1 but, for example, [`conditional_rand`](@ref)
returns a `Real` rather than an `AbstractVector` of length 1.
"""
struct ScalarOutputLGC{
    TA<:Adjoint{<:Any, <:AbstractVector}, Ta<:Real, TQ<:Real,
} <: AbstractLGC
    A::TA
    a::Ta
    Q::TQ
end

dim_out(f::ScalarOutputLGC) = 1

dim_in(f::ScalarOutputLGC) = size(f.A, 2)

get_fields(f::ScalarOutputLGC) = (f.A, f.a, f.Q)

noise_cov(f::ScalarOutputLGC) = f.Q

function conditional_rand(ε::Real, f::ScalarOutputLGC, x::AbstractVector)
    return (f.A * x + f.a) + sqrt(f.Q) * ε
end

ε_randn(rng::AbstractRNG, f::ScalarOutputLGC) = randn(rng, eltype(f))

function posterior_and_lml(x::Gaussian, f::ScalarOutputLGC, y::T) where {T<:Real}
    m, P = get_fields(x)
    A, a, Q = get_fields(f)
    V = A * P
    sqrtS = sqrt(V * A' + Q)
    B = sqrtS \ V
    α = sqrtS \ (y - (A * m + a))

    lml = -(convert(T, log(2π)) + 2 * log(sqrtS) + α^2) / 2
    return Gaussian(m + B'α, P - B'B), lml
end



"""
    BottleneckLGC

A composition of an affine map that projects onto a low-dimensional subspace and a
[`LargeOutputLGC`](@ref). This structure is exploited by only ever computing `Cholesky`
factorisations in the space the affine map maps to, rather than the input or output space.

Letting, `H` and `h` parametrise the affine map, and `f` the "fan-out" [`LargeOutputLGC`](@ref), the
conditional distribution that this model parametrises is
```julia
y | x ~ Gaussian(f.A * (H * x + h) + f.a, f.Q)
```

Note that this type does not enforce that `size(H, 1) < size(H, 2)`, nor that
`dim_out(f) > dim_in(f)`, it's just not a particularly good idea to use this type unless
your model satisfies these properties.
"""
struct BottleneckLGC{
    TH<:AbstractMatrix{<:Real}, Th<:AbstractVector{<:Real}, Tout<:LargeOutputLGC,
} <: AbstractLGC
    H::TH
    h::Th
    fan_out::Tout
end

function Base.:(==)(x::BottleneckLGC, y::BottleneckLGC)
    return (x.H == y.H) && (x.h == y.h) && (x.fan_out == y.fan_out)
end

Base.eltype(f::BottleneckLGC) = eltype(f.fan_out)

dim_out(f::BottleneckLGC) = dim_out(f.fan_out)

dim_in(f::BottleneckLGC) = size(f.H, 2)

noise_cov(f::BottleneckLGC) = noise_cov(f.fan_out)

get_fields(f::BottleneckLGC) = (f.H, f.h, f.fan_out)

function conditional_rand(ε::AbstractVector{<:Real}, f::BottleneckLGC, x::AbstractVector)
    H, h, fan_out = get_fields(f)
    return conditional_rand(ε, fan_out, H * x + h)
end

ε_randn(rng::AbstractRNG, f::BottleneckLGC) = ε_randn(rng, f.fan_out)

# Construct the low-dimensional projection given by f.H and f.h of `x`.
function _project(x::Gaussian, f::BottleneckLGC)
    m, P = get_fields(x)
    H, h, _ = get_fields(f)
    return Gaussian(H * m + h, H * P * H' + ident_eps(x))
end

predict(x::Gaussian, f::BottleneckLGC) = predict(_project(x, f), f.fan_out)

function predict_marginals(x::Gaussian, f::BottleneckLGC)
    return predict_marginals(_project(x, f), f.fan_out)
end

function posterior_and_lml(x::Gaussian, f::BottleneckLGC, y::AbstractVector)

    xm, xP = get_fields(x)
    H, _, fan_out = get_fields(f)

    # Get the posterior over the intermediate variable `z`.
    z = _project(x, f)
    z_post, lml = posterior_and_lml(z, fan_out, y)

    # Compute the posterior `x | y` by integrating `x | z` against `z | y`.
    zm, zP = get_fields(z)
    z_postm, z_postP = get_fields(z_post)
    U = cholesky(symmetric(zP + ident_eps(z, 1e-12))).U
    Gt = U \ (U' \ (H * xP))
    return Gaussian(xm + Gt' * (z_postm - zm), xP + Gt' * (z_postP - zP) * Gt), lml
end
