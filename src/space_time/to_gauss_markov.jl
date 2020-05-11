using Kronecker: KroneckerProduct

my_I(T, N) = Matrix{T}(I, N, N)
Zygote.@nograd my_I

function GaussMarkovModel(k::Separable, x::SpaceTimeGrid, storage)

    # Compute spatial covariance, and temporal GaussMarkovModel.
    r, t = x.xl, x.xr
    kr, kt = k.l, k.r
    Kr = pw(kr, r)
    gmm_time = GaussMarkovModel(kt, t, storage)

    # Produce a new GaussMarkovModel over the spatial locations specified.
    Nr = length(r)
    ident = my_I(eltype(storage), Nr)
    # A = kron.(Ref(Eye(Nr)), gmm_time.A)
    A = map(A -> kron(ident, A), gmm_time.A)
    a = map(a -> repeat(a, Nr), gmm_time.a)
    # a = repeat.(gmm_time.a, Nr)
    Q = map(Q -> kron(Kr, Q), gmm_time.Q)
    H = map(H -> kron(ident, H), gmm_time.H)
    # H = build_H(ident, gmm_time.H)
    # h = repeat.(gmm_time.h, Nr)
    h = map(h -> repeat(h, Nr), gmm_time.h)
    x = Gaussian(
        repeat(gmm_time.x0.m, Nr),
        kron(Kr, gmm_time.x0.P),
    )
    return GaussMarkovModel(A, a, Q, H, h, x)
end

function build_H(ident, Hs::Fill)
    return Fill(kron(ident, Hs.value), length(Hs))
end

function build_H(ident, Hs::AbstractVector)
    return kron.(Ref(ident), gmm_time.H)
end

function (f::LTISDE)(x::SpaceTimeGrid, Σs::AV{<:AM{<:Real}})
    return LGSSM(GaussMarkovModel(f.f.k, x, f.storage), Σs)
end

function (f::LTISDE)(x::SpaceTimeGrid, σ²::Real)
    Σ = Diagonal(Fill(σ², length(x.xl)))
    Σs = Fill(collect(Σ), length(x.xr))
    return f(x, Σs)
end

(f::LTISDE)(x::SpaceTimeGrid) = f(x, 0.0)
