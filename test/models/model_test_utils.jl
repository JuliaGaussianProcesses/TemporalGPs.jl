using TemporalGPs:
    GaussMarkovModel,
    dim_latent,
    dim_obs,
    LGSSM,
    ScalarLGSSM,
    Gaussian,
    StorageType,
    ScalarStorage,
    is_of_storage_type,
    storage_type,
    LinearGaussianDynamics

#
# Generation of positive semi-definite matrices.
#

function random_vector(rng::AbstractRNG, N::Int, ::ArrayStorage{T}) where {T<:Real}
    return randn(rng, T, N)
end

function random_vector(rng::AbstractRNG, N::Int, ::SArrayStorage{T}) where {T<:Real}
    return SVector{N}(randn(rng, T, N))
end

function random_matrix(rng::AbstractRNG, M::Int, N::Int, ::ArrayStorage{T}) where {T<:Real}
    return randn(rng, T, M, N)
end

function random_matrix(rng::AbstractRNG, M::Int, N::Int, ::SArrayStorage{T}) where {T<:Real}
    return SMatrix{M, N}(randn(rng, T, M, N))
end

function random_nice_psd_matrix(rng::AbstractRNG, N::Integer, ::ArrayStorage{T}) where {T}

    # Generate random positive definite matrix.
    U = UpperTriangular(randn(rng, T, N, N))
    S = U'U + T(1e-1) * I
    S = Symmetric(pw(Matern12(), 5 .* randn(rng, T, N)) + T(1e-3) * I)

    # Centre (make eigenvals N(0, 2^2)) and bound the eigenvalues between 0 and 1.
    λ, Γ = eigen(S)
    m_λ = N > 1 ? mean(λ) : mean(λ) + T(0.1)
    σ_λ = N > 1 ? std(λ) : T(1.0)
    λ .= 2 .* (λ .- (m_λ + 0.1)) ./ σ_λ
    @. λ = T(1 / (1 + exp(-λ)) * 0.9 + 0.1)
    return collect(Symmetric(Γ * Diagonal(λ) * Γ'))
end

function random_nice_psd_matrix(rng::AbstractRNG, N::Integer, ::SArrayStorage{T}) where {T}
    return SMatrix{N, N, T}(random_nice_psd_matrix(rng, N, ArrayStorage(T)))
end

function random_nice_psd_matrix(rng::AbstractRNG, ::Integer, ::ScalarStorage{T}) where {T}
    return rand(rng, T) + convert(T, 0.1)
end



#
# Generation of Gaussians.
#

function random_gaussian(rng::AbstractRNG, dim::Int, s::StorageType)
    return Gaussian(random_vector(rng, dim, s), random_nice_psd_matrix(rng, dim, s))
end

function FiniteDifferences.rand_tangent(rng::AbstractRNG, d::T) where {T<:Gaussian}
    return Composite{T}(
        m=rand_tangent(rng, d.m),
        P=random_nice_psd_matrix(rng, length(d.m), storage_type(d)),
    )
end



#
# Generation of LinearGaussianDynamics.
#

function random_linear_gaussian_dynamics(
    rng::AbstractRNG, Dlat::Int, Dobs::Int, s::StorageType,
)
    return LinearGaussianDynamics(
        random_matrix(rng, Dobs, Dlat, s),
        random_vector(rng, Dobs, s),
        random_nice_psd_matrix(rng, Dobs, s),
    )
end



#
# Generation of GaussMarkovModels.
#

function random_tv_gmm(
    rng::AbstractRNG, ordering, Dlat::Int, Dobs::Int, N::Int, s::StorageType,
)

    As = map(_ -> -random_nice_psd_matrix(rng, Dlat, s), 1:N)
    as = map(_ -> random_vector(rng, Dlat, s), 1:N)
    x0 = random_gaussian(rng, Dlat, s)

    # For some reason this operation seems to be _incredibly_ inaccurate. My guess is that
    # the numerics are horrible for some reason, but I'm not sure why. Hence we add a pretty
    # large constant to the diagonal to ensure that all Qs are positive definite.
    Qs = map(n -> x0.P - Symmetric(As[n] * x0.P * As[n]') + eltype(s)(1e-1) * I, 1:N)

    return GaussMarkovModel(ordering, As, as, Qs, x0)
end

function random_ti_gmm(
    rng::AbstractRNG, ordering, Dlat::Int, Dobs::Int, N::Int, s::StorageType,
)

    As = Fill(-random_nice_psd_matrix(rng, Dlat, s), N)
    as = Fill(random_vector(rng, Dlat, s), N)
    x0 = random_gaussian(rng, Dlat, s)

    # For some reason this operation seems to be _incredibly_ inaccurate. My guess is that
    # the numerics are horrible for some reason, but I'm not sure why. Hence we add a pretty
    # large constant to the diagonal to ensure that all Qs are positive definite.
    Qs = Fill(x0.P - As[1] * x0.P * As[1]' + eltype(s)(1e-1) * I, N)
    return GaussMarkovModel(ordering, As, as, Qs, x0)
end

function FiniteDifferences.rand_tangent(rng::AbstractRNG, gmm::T) where {T<:GaussMarkovModel}
    return Composite{T}(
        ordering = nothing,
        As = rand_tangent(rng, gmm.As),
        as = rand_tangent(rng, gmm.as),
        Qs = map(Q -> random_nice_psd_matrix(rng, size(Q, 1), storage_type(gmm)), gmm.Qs),
        x0 = rand_tangent(rng, gmm.x0),
    )
end


#
# Generation of LGSSMs.
#

function random_tv_lgssm(rng::AbstractRNG, ordering, Dlat::Int, Dobs::Int, N::Int, storage)
    gmm = random_tv_gmm(rng, ordering, Dlat, Dobs, N, storage)
    Hs = map(_ -> random_matrix(rng, Dobs, Dlat, storage), 1:N)
    hs = map(_ -> random_vector(rng, Dobs, storage), 1:N)
    Σs = map(_ -> random_nice_psd_matrix(rng, Dobs, storage), 1:N)
    return LGSSM(gmm, Hs, hs, Σs)
end

function random_ti_lgssm(rng::AbstractRNG, ordering, Dlat::Int, Dobs::Int, N::Int, storage)
    gmm = random_ti_gmm(rng, ordering, Dlat, Dobs, N, storage)
    Hs = Fill(random_matrix(rng, Dobs, Dlat, storage), N)
    hs = Fill(random_vector(rng, Dobs, storage), N)
    Σs = Fill(random_nice_psd_matrix(rng, Dobs, storage), N)
    return LGSSM(gmm, Hs, hs, Σs)
end

function FiniteDifferences.rand_tangent(rng::AbstractRNG, ssm::T) where {T<:LGSSM}
    return Composite{T}(
        gmm = rand_tangent(rng, ssm.gmm),
        Hs = rand_tangent(rng, ssm.Hs),
        hs = rand_tangent(rng, ssm.hs),
        Σs = map(
            _ -> random_nice_psd_matrix(rng, dim_obs(ssm), storage_type(ssm)),
            ssm.Σs,
        ),
    )
end

function random_tv_scalar_lgssm(rng::AbstractRNG, Dlat::Int, N::Int, storage)
    return ScalarLGSSM(random_tv_lgssm(rng, Dlat, 1, N, storage))
end

function random_ti_scalar_lgssm(rng::AbstractRNG, Dlat::Int, N::Int, storage)
    return ScalarLGSSM(random_ti_lgssm(rng, Dlat, 1, N, storage))
end

function random_tv_posterior_lgssm(rng::AbstractRNG, Dlat::Int, Dobs::Int, N::Int, storage)
    lgssm = random_tv_lgssm(rng, Dlat, Dobs, N, storage)
    y = rand(rng, lgssm)
    Σs = map(_ -> random_nice_psd_matrix(rng, Dobs, storage), eachindex(y))
    return posterior(lgssm, y, Σs)
end

function random_ti_posterior_lgssm(rng::AbstractRNG, Dlat::Int, Dobs::Int, N::Int, storage)
    lgssm = random_ti_lgssm(rng, Dlat, Dobs, N, storage)
    y = rand(rng, lgssm)
    Σs = Fill(random_nice_psd_matrix(rng, Dobs, storage), length(lgssm))
    return posterior(lgssm, y, Σs)
end


#
# Validation of internal consistency.
#

function validate_dims(gmm::GaussMarkovModel)

    # Check all vectors are the correct length.
    @test length(gmm.A) == length(gmm)
    @test length(gmm.a) == length(gmm)
    @test length(gmm.Q) == length(gmm)
    @test length(gmm.H) == length(gmm)
    @test length(gmm.h) == length(gmm)

    # Check sizes of each element of the struct are correct.
    N = length(gmm)
    Dlat = dim_latent(gmm)
    Dobs = dim_obs(gmm)
    @test all(map(n -> size(gmm.A[n]) == (Dlat, Dlat), 1:N))
    @test all(map(n -> size(gmm.a[n]) == (Dlat,), 1:N))
    @test all(map(n -> size(gmm.Q[n]) == (Dlat, Dlat), 1:N))
    @test all(map(n -> size(gmm.H[n]) == (Dobs, Dlat), 1:N))
    @test all(map(n -> size(gmm.h[n]) == (Dobs,), 1:N))
    @test size(gmm.x0.m) == (Dlat,)
    @test size(gmm.x0.P) == (Dlat, Dlat)
    return nothing
end

function validate_dims(model::LGSSM)
    validate_dims(model.gmm)

    N = length(model)
    Dobs = dim_obs(model.gmm)
    @test all(map(n -> size(model.Σ[n]) == (Dobs, Dobs), 1:N))
    return nothing
end

function validate_dims(model::ScalarLGSSM)
    validate_dims(model.model)
    return nothing
end

function __verify_model_properties(model, Dlat, Dobs, N, storage_type)
    @test is_of_storage_type(model, storage_type)
    @test length(model) == N
    @test dim_obs(model) == Dobs
    @test dim_latent(model) == Dlat
    validate_dims(model)
end

function __verify_model_properties(model, Dlat, N, storage_type)
    return __verify_model_properties(model, Dlat, 1, N, storage_type)
end
