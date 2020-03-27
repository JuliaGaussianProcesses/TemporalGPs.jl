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
    return GaussMarkovModel(
        collect.(KroneckerProduct.(Ref(Eye(Nr)), gmm_time.A)),
        repeat.(gmm_time.a, Nr),
        collect.(KroneckerProduct.(Ref(Kr), gmm_time.Q)),
        collect.(KroneckerProduct.(Ref(Eye(Nr)), gmm_time.H)),
        repeat.(gmm_time.h, Nr),
        Gaussian(
            repeat(gmm_time.x0.m, Nr),
            collect(Kr ⊗ gmm_time.x0.P),
        ),
        # map(n -> collect(Eye(Nr) ⊗ gmm_time.A[n]), 1:Nt),
        # map(n -> repeat(gmm_time.a[n], Nr), 1:Nt),
        # map(n -> collect(Kr ⊗ gmm_time.Q[n]), 1:Nt),
        # map(n -> collect(Eye(Nr) ⊗ gmm_time.H[n]), 1:Nt),
        # map(n -> repeat(gmm_time.h[n], Nr), 1:Nt),
        # Gaussian(
        #     repeat(gmm_time.x0.m, Nr),
        #     collect(Kr ⊗ gmm_time.x0.P),
        # ),
    )
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
