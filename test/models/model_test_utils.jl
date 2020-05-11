using TemporalGPs: GaussMarkovModel, dim_latent, dim_obs, LGSSM, ScalarLGSSM, Gaussian,
    StorageType



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
    S = U'U + T(1e-2) * I
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

@testset "random_nice_psd_matrix" begin
    rng = MersenneTwister(123456)
    storages = [
        (name="dense storage", val=ArrayStorage(Float64)),
        (name="static storage", val=SArrayStorage(Float64)),
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

function random_gaussian(rng::AbstractRNG, dim::Int, s::ArrayStorage{T}) where {T<:Real}
    return Gaussian(randn(rng, T, dim), Symmetric(random_nice_psd_matrix(rng, dim, s)))
end

function random_gaussian(rng::AbstractRNG, dim::Int, s::SArrayStorage{T}) where {T<:Real}
    return to_static(random_gaussian(rng, dim, ArrayStorage{T}))
end

function random_tv_gmm(
    rng::AbstractRNG, Dlat::Int, Dobs::Int, N::Int, s::ArrayStorage{T},
) where {T<:Real}
    As = map(_ -> -random_nice_psd_matrix(rng, Dlat, s), 1:N)
    as = map(_ -> randn(rng, T, Dlat), 1:N)
    Hs = map(_ -> randn(rng, T, Dobs, Dlat), 1:N)
    hs = map(_ -> randn(rng, T, Dobs), 1:N)
    x0 = TemporalGPs.Gaussian(
        randn(rng, T, Dlat),
        random_nice_psd_matrix(rng, Dlat, s),
    )

    # For some reason this operation seems to be _incredibly_ inaccurate. My guess is that
    # the numerics are horrible for some reason, but I'm not sure why. Hence we add a pretty
    # large constant to the diagonal to ensure that all Qs are positive definite.
    Qs = map(n -> x0.P - Symmetric(As[n] * x0.P * As[n]') + T(1e-1) * I, 1:N)

    return GaussMarkovModel(As, as, Qs, Hs, hs, x0)
end

function random_tv_gmm(
    rng::AbstractRNG, Dlat::Int, Dobs::Int, N::Int, s::SArrayStorage{T},
) where {T<:Real}
    return to_static(random_tv_gmm(rng, Dlat, Dobs, N, ArrayStorage(T)))
end

function random_ti_gmm(
    rng::AbstractRNG, Dlat::Int, Dobs::Int, N::Int, s::ArrayStorage{T},
) where {T<:Real}
    As = Fill(-random_nice_psd_matrix(rng, Dlat, s), N)
    as = Fill(randn(rng, T, Dlat), N)
    Hs = Fill(randn(rng, T, Dobs, Dlat), N)
    hs = Fill(randn(rng, T, Dobs), N)
    x0 = TemporalGPs.Gaussian(
        randn(rng, T, Dlat),
        random_nice_psd_matrix(rng, Dlat, s),
    )

    # For some reason this operation seems to be _incredibly_ inaccurate. My guess is that
    # the numerics are horrible for some reason, but I'm not sure why. Hence we add a pretty
    # large constant to the diagonal to ensure that all Qs are positive definite.
    Qs = Fill(x0.P - As[1] * x0.P * As[1]' + T(1e-1) * I, N)
    return GaussMarkovModel(As, as, Qs, Hs, hs, x0)
end

function random_ti_gmm(
    rng::AbstractRNG, Dlat::Int, Dobs::Int, N::Int, s::SArrayStorage{T},
) where {T<:Real}
    return to_static(random_ti_gmm(rng, Dlat, Dobs, N, ArrayStorage(T)))
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



#
# Validate that all parameter containers are of the expected types.
#

validate_types(::Any, ::StorageType) = @test false

validate_types(::T, ::StorageType{T}) where {T<:Real} = @test true

validate_types(::Vector{T}, ::ArrayStorage{T}) where {T} = @test true

validate_types(::Matrix{T}, ::ArrayStorage{T}) where {T} = @test true

validate_types(::Zeros{T}, ::ArrayStorage{T}) where {T} = @test true

validate_types(::SVector{D, T} where {D}, ::SArrayStorage{T}) where {T} = @test true

validate_types(::SMatrix{D1, D2, T} where {D1, D2}, ::SArrayStorage{T}) where {T} = @test true

validate_types(::Zeros{T}, ::SArrayStorage{T}) where {T} = @test true

function validate_types(ft::GaussMarkovModel, s::StorageType)
    validate_types(first(ft.A), s)
    validate_types(first(ft.a), s)
    validate_types(first(ft.Q), s)
    validate_types(first(ft.H), s)
    validate_types(first(ft.h), s)
    validate_types(ft.x0, s)
end

function validate_types(x::Gaussian, s::StorageType)
    validate_types(x.m, s)
    validate_types(x.P, s)
end
