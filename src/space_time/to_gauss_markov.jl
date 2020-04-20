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
    # A = kron.(Ref(Eye(Nr)), gmm_time.A)
    A = map(A -> Eye(Nr) ⊗ A, gmm_time.A)
    a = repeat.(gmm_time.a, Nr)
    # a = map(n -> repeat(gmm_time.a[n], Nr), 1:Nt)
    # Q = kron.(Ref(Kr), gmm_time.Q)
    Q = map(Q -> kron(Kr, Q), gmm_time.Q)
    H = kron.(Ref(Eye(Nr)), gmm_time.H)
    h = repeat.(gmm_time.h, Nr)
    x = Gaussian(
        repeat(gmm_time.x0.m, Nr),
        kron(Kr, gmm_time.x0.P),
    )
    return GaussMarkovModel(
        Zygote.hook(Δ->(@show typeof(Δ), size(Δ); Δ), A),
        Zygote.hook(Δ->(@show typeof(Δ), size(Δ); Δ), a),
        Zygote.hook(Δ->(@show typeof(Δ), size(Δ); Δ), Q),
        Zygote.hook(Δ->(@show typeof(Δ), size(Δ); Δ), H),
        Zygote.hook(Δ->(@show typeof(Δ), size(Δ); Δ), h),
        x,
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


function GaussMarkovModel(k::Stheno.Sum, ts::SpaceTimeGrid, storage_type)
    model_l = GaussMarkovModel(k.kl, ts, storage_type)
    model_r = GaussMarkovModel(k.kr, ts, storage_type)

    return GaussMarkovModel(
        sparse_blk_diag.(model_l.A, model_r.A),
        vcat.(model_l.a, model_r.a),
        sparse_blk_diag.(model_l.Q, model_r.Q),
        hcat.(model_l.H, model_r.H),
        model_l.h + model_r.h,
        Gaussian(
            collect(vcat(model_l.x0.m, model_r.x0.m)),
            blk_diag(model_l.x0.P, model_r.x0.P),
        ),
    )
end

sparse_blk_diag(A::AbstractMatrix, B::AbstractMatrix) = BlockDiagonal([A, B])

function sparse_blk_diag(A::AbstractMatrix, B::BlockDiagonal)
    return BlockDiagonal(vcat([A], B.blocks))
end

function sparse_blk_diag(A::BlockDiagonal, B::AbstractMatrix)
    return BlockDiagonal(vcat(A.blocks, [B]))
end

function sparse_blk_diag(A::BlockDiagonal, B::BlockDiagonal)
    return BlockDiagonal(vcat(A.blocks, B.blocks))
end
