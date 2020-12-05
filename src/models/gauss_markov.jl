"""
    GaussMarkovModel

Specifies a Gauss-Markov model. The transformation of it that you get to observe is
specified by `H` and `h`.
```julia
x[0] ∼ x0
x[t] = A[t] * x[t-1] + a[t] + ε[t], ε[t] ∼ N(0, Q)
f[t] = H[t] * x[t] + h[t]
```
"""
struct GaussMarkovModel{
    TA<:AV{<:AM{<:Real}},
    Ta<:AV{<:AV{<:Real}},
    TQ<:AV{<:AM{<:Real}},
    TH<:AV{<:AM{<:Real}},
    Th<:AV{<:AV{<:Real}},
    Tx0<:Gaussian,
}
    A::TA
    a::Ta
    Q::TQ
    H::TH
    h::Th
    x0::Tx0
end

function Base.eltype(
    ::GaussMarkovModel{<:AbstractVector{TA}},
) where {T<:Real, TA<:AbstractMatrix{T}}
    return T
end

Base.length(ft::GaussMarkovModel) = length(ft.A)

function Base.getindex(ft::GaussMarkovModel, n::Int)
    return (
        A = ft.A[n],
        a = ft.a[n],
        Q = ft.Q[n],
        H = ft.H[n],
        h = ft.h[n],
    )
end

function Base.:(==)(x::GaussMarkovModel, y::GaussMarkovModel)
    return (x.A == y.A) && (x.a == y.a) && (x.Q == y.Q) && (x.H == y.H) &&
        (x.h == y.h) && (x.x0 == y.x0)
end

dim_obs(ft::GaussMarkovModel) = size(first(ft.H), 1)

dim_latent(ft::GaussMarkovModel) = size(first(ft.H), 2)

storage_type(gmm::GaussMarkovModel{<:AV{Matrix{T}}}) where {T<:Real} = ArrayStorage(T)

function storage_type(gmm::GaussMarkovModel{<:AV{<:SMatrix{D, D, T}}}) where {D, T<:Real}
    return SArrayStorage(T)
end

function is_of_storage_type(gmm::GaussMarkovModel, s::StorageType)
    return is_of_storage_type((gmm.A, gmm.a, gmm.Q, gmm.H, gmm.h, gmm.x0), s)
end

is_time_invariant(gmm::GaussMarkovModel) = false
is_time_invariant(gmm::GaussMarkovModel{<:Fill, <:Fill, <:Fill, <:Fill, <:Fill}) = true


"""
    mean(gmm::GaussMarkovModel)

Compute the mean vector of `gmm`. Warning: this method is indented for testing purposes,
not for efficiently computing things in practice. For that, see `correlate`, `decorrelate`,
and related functions.
"""
function mean(gmm::GaussMarkovModel)

    # Pull out parameters to reduce visual clutter.
    As = gmm.A
    as = gmm.a
    Hs = gmm.H
    hs = gmm.h
    m0 = gmm.x0.m

    # Compute mean of the first state.
    mx = As[1] * m0 + as[1]
    mf_1 = Hs[1] * mx + hs[1]

    # Allocate memory assuming type-stability.
    mfs = Vector{typeof(mf_1)}(undef, length(As))
    mfs[1] = mf_1

    # Iterate through the remainder.
    for n in 2:length(As)
        mx = As[n] * mx + as[n]
        mfs[n] = Hs[n] * mx + hs[n]
    end

    # Concatenate the result into a single vector.
    return vcat(mfs...)
end

"""
    cov(gmm::GaussMarkovModel)

Compute the covariance matrix of `gmm`. Warning: this method is indented for testing
purposes, not for efficiently computing things in practice. For that, see `correlate`,
`decorrelate`, and related functions.
"""
function cov(gmm::GaussMarkovModel)

    # Pull out parameters to reduce visual clutter.
    As = gmm.A
    Qs = gmm.Q
    Hs = gmm.H
    P = gmm.x0.P

    # Compute the top-left block of the covariance matrix.
    Px_11 = As[1] * P * As[1]' + Qs[1]
    Pf_11 = Hs[1] * Px_11 * Hs[1]'

    # Allocate memory for the rest assuming type-stability.
    Pxs = Matrix{typeof(Px_11)}(undef, length(As), length(As))
    Pxs[1, 1] = Px_11

    Pfs = Matrix{typeof(Pf_11)}(undef, length(As), length(As))
    Pfs[1, 1] = Pf_11

    # Fill out the rest of the first col of the matrix. Copy into first row since symmetric.
    for m in 2:length(As)
        Pxs[m, 1] = As[m] * Pxs[m - 1, 1]
        Pxs[1, m] = collect(Pxs[m, 1]')
        Pfs[m, 1] = Hs[m] * Pxs[m, 1] * Hs[1]'
        Pfs[1, m] = collect(Pfs[m, 1]')
    end

    # Iterate through the rest of the rest of the matrix. The result is symmetric, so the
    # upper triangle is simply copied into the lower.
    for n in 2:length(As)
        Pxs[n, n] = As[n] * Pxs[n - 1, n - 1] * As[n]' + Qs[n]
        Pfs[n, n] = Hs[n] * Pxs[n, n] * Hs[n]'

        for m in (n + 1):length(As)
            Pxs[m, n] = As[m] * Pxs[m - 1, n]
            Pxs[n, m] = collect(Pxs[m, n]')
            Pfs[m, n] = Hs[m] * Pxs[m, n] * Hs[n]'
            Pfs[n, m] = collect(Pfs[m, n]')
        end
    end

    return Array(BlockArrays.mortar(Pfs))
end
