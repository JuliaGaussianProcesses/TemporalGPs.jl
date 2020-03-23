using TemporalGPs: GaussMarkovModel, DenseStorage, StaticStorage, dim_latent, dim_obs,
    LGSSM, ScalarLGSSM, Gaussian



#
# Convert things containing arrays into things containing StaticArrays.
#

to_static(x::AbstractVector{<:Number}) = SVector{length(x)}(x)

to_static(X::AbstractMatrix{<:Number}) = SMatrix{size(X, 1), size(X, 2)}(X)

to_static(x::Gaussian) = Gaussian(to_static(x.m), to_static(x.P))

function to_static(gmm::GaussMarkovModel)
    return GaussMarkovModel(
        to_static.(gmm.A),
        to_static.(gmm.a),
        to_static.(gmm.Q),
        to_static.(gmm.H),
        to_static.(gmm.h),
        TemporalGPs.Gaussian(to_static(gmm.x0.m), to_static(gmm.x0.P)),
    )
end

to_static(model::LGSSM) = LGSSM(to_static(model.gmm), to_static.(model.Σ))



#
# Generation of positive semi-definite matrices.
#

function random_nice_psd_matrix(rng::AbstractRNG, N::Integer, ::DenseStorage)

    # Generate random positive definite matrix.
    U = UpperTriangular(randn(rng, N, N))
    S = U'U + 1e-2I
    S = pw(Matern12(), 5 .* randn(rng, N)) + 1e-3I

    # Centre (make eigenvals N(0, 2^2)) and bound the eigenvalues between 0 and 1.
    λ, Γ = eigen(S)
    m_λ = N > 1 ? mean(λ) : mean(λ) + 0.1
    σ_λ = N > 1 ? std(λ) : 1.0
    λ .= 2 .* (λ .- (m_λ + 0.1)) ./ σ_λ
    @. λ = 1 / (1 + exp(-λ)) * 0.9 + 0.1
    return collect(Symmetric(Γ * Diagonal(λ) * Γ'))
end

function random_nice_psd_matrix(rng::AbstractRNG, N::Integer, ::StaticStorage)
    return SMatrix{N, N}(random_nice_psd_matrix(rng, N, DenseStorage()))
end

@testset "random_nice_psd_matrix" begin
    rng = MersenneTwister(123456)
    storages = [
        (name="dense storage", val=DenseStorage()),
        (name="static storage", val=StaticStorage()),
    ]

    @testset "$(storage.name)" for storage in storages
        Σ = random_nice_psd_matrix(rng, 11, storage.val)
        @test all(eigvals(Σ) .> 0)
        @test all(eigvals(Σ) .< 1)
    end
end



#
# Generation of GaussMarkovModels.
#

function random_tv_gmm(rng::AbstractRNG, Dlat::Int, Dobs::Int, N::Int, ::DenseStorage)
    As = map(_ -> -random_nice_psd_matrix(rng, Dlat, DenseStorage()), 1:N)
    as = map(_ -> randn(rng, Dlat), 1:N)
    Hs = map(_ -> randn(rng, Dobs, Dlat), 1:N)
    hs = map(_ -> randn(rng, Dobs), 1:N)
    x0 = TemporalGPs.Gaussian(
        randn(rng, Dlat),
        random_nice_psd_matrix(rng, Dlat, DenseStorage()),
    )

    # For some reason this operation seems to be _incredibly_ inaccurate. My guess is that
    # the numerics are horrible for some reason, but I'm not sure why. Hence we add a pretty
    # large constant to the diagonal to ensure that all Qs are positive definite.
    Qs = map(n -> x0.P - Symmetric(As[n] * x0.P * As[n]') + 1e-1I, 1:N)

    return GaussMarkovModel(As, as, Qs, Hs, hs, x0)
end

function random_tv_gmm(rng::AbstractRNG, Dlat::Int, Dobs::Int, N::Int, ::StaticStorage)
    return to_static(random_tv_gmm(rng, Dlat, Dobs, N, DenseStorage()))
end

function random_ti_gmm(rng::AbstractRNG, Dlat::Int, Dobs::Int, N::Int, ::DenseStorage)
    As = Fill(-random_nice_psd_matrix(rng, Dlat, DenseStorage()), N)
    as = Fill(randn(rng, Dlat), N)
    Hs = Fill(randn(rng, Dobs, Dlat), N)
    hs = Fill(randn(rng, Dobs), N)
    x0 = TemporalGPs.Gaussian(
        randn(rng, Dlat),
        random_nice_psd_matrix(rng, Dlat, DenseStorage()),
    )

    # For some reason this operation seems to be _incredibly_ inaccurate. My guess is that
    # the numerics are horrible for some reason, but I'm not sure why. Hence we add a pretty
    # large constant to the diagonal to ensure that all Qs are positive definite.
    Qs = Fill(x0.P - As[1] * x0.P * As[1]' + 1e-1I, N)
    return GaussMarkovModel(As, as, Qs, Hs, hs, x0)
end

function random_ti_gmm(rng::AbstractRNG, Dlat::Int, Dobs::Int, N::Int, ::StaticStorage)
    return to_static(random_ti_gmm(rng, Dlat, Dobs, N, DenseStorage()))
end



#
# Generation of LGSSMs.
#

function random_tv_lgssm(rng::AbstractRNG, Dlat::Int, Dobs::Int, N::Int, storage)
    gmm = random_tv_gmm(rng, Dlat, Dobs, N, storage)
    Σ = map(_ -> random_nice_psd_matrix(rng, Dobs, storage), 1:N)
    return LGSSM(gmm, Σ)
end

function random_ti_lgssm(rng::AbstractRNG, Dlat::Int, Dobs::Int, N::Int, storage)
    gmm = random_ti_gmm(rng, Dlat, Dobs, N, storage)
    Σ = Fill(random_nice_psd_matrix(rng, Dobs, storage), N)
    return LGSSM(gmm, Σ)
end

function random_tv_scalar_lgssm(rng::AbstractRNG, Dlat::Int, N::Int, storage)
    return ScalarLGSSM(random_tv_lgssm(rng, Dlat, 1, N, storage))
end

function random_ti_scalar_lgssm(rng::AbstractRNG, Dlat::Int, N::Int, storage)
    return ScalarLGSSM(random_ti_lgssm(rng, Dlat, 1, N, storage))
end
