using Kronecker: KroneckerProduct

function GaussMarkovModel(k::Separable, x::SpaceTimeGrid, storage)

    # Compute spatial covariance, and temporal GaussMarkovModel.
    r, t = x.xl, x.xr
    kr, kt = k.l, k.r
    Kr = pw(kr, r)
    gmm_time = GaussMarkovModel(kt, t, storage)

    # Produce a new GaussMarkovModel over the spatial locations specified.
    Nr = length(r)
    Nt = length(t)
    A = map(A -> kron(Eye(Nr), A), gmm_time.A)
    a = repeat.(gmm_time.a, Nr)
    Q = map(Q -> kron(Kr, Q), gmm_time.Q)
    H = kron.(Ref(Eye(Nr)), gmm_time.H)
    h = repeat.(gmm_time.h, Nr)
    x = Gaussian(
        repeat(gmm_time.x0.m, Nr),
        kron(Kr, gmm_time.x0.P),
    )
    return GaussMarkovModel(A, a, Q, H, h, x)
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
