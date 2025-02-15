using TemporalGPs:
    ArrayStorage,
    SArrayStorage,
    GaussMarkovModel,
    Forward,
    Reverse,
    dim,
    LGSSM,
    Gaussian,
    StorageType,
    ScalarStorage,
    is_of_storage_type,
    storage_type,
    SmallOutputLGC,
    ScalarOutputLGC,
    LargeOutputLGC,
    BottleneckLGC

# Generation of positive semi-definite matrices.

function random_vector(rng::AbstractRNG, N::Int, ::ArrayStorage{T}) where {T<:Real}
    return randn(rng, T, N)
end

function random_vector(rng::AbstractRNG, N::Int, ::SArrayStorage{T}) where {T<:Real}
    return SVector{N}(randn(rng, T, N))
end

function random_matrix(rng::AbstractRNG, M::Int, N::Int, ::ArrayStorage{T}) where {T<:Real}
    return diagm(M, N, ones(T, min(M, N))) .+ T(1e-1) .* randn(rng, T, M, N)
end

function random_matrix(rng::AbstractRNG, M::Int, N::Int, ::SArrayStorage{T}) where {T<:Real}
    return SMatrix{M, N}(random_matrix(rng, M, N, ArrayStorage{T}()))
end

function random_psd_matrix(
    rng::AbstractRNG,
    N::Int,
    eig_lb::Float64,
    eig_ub::Float64,
    ::Val{:dense},
    ::ArrayStorage{T},
) where {T}

    # Matrix not guaranteed to be positive definite if eigenvalues can be less than 0.
    @assert eig_lb > 0

    # Generate a random orthogonal matrix by computing the QR decomposition of a random
    # matrix, and extracting the "Q" component.
    Q = collect(qr(randn(rng, N, N)).Q)

    # Generate eigenvalues in the desired range.
    λs = rand(rng, N) .* T(eig_ub - eig_lb) .+ T(eig_lb)

    # Construct a positive definite matrix.
    return collect(Symmetric(Q * Diagonal(λs) * Q'))

    # # Centre (make eigenvals N(0, 2^2)) and bound the eigenvalues between 0 and 1.
    # λ, Γ = eigen(S)
    # m_λ = N > 1 ? mean(λ) : mean(λ) + T(1.0)
    # σ_λ = N > 1 ? std(λ) : T(1.0)
    # λ .= 2 .* (λ .- (m_λ + 0.1)) ./ σ_λ
    # @. λ = T(1 / (1 + exp(-λ)) * 0.9 + 0.1)
    # return collect(Symmetric(Γ * Diagonal(λ) * Γ'))
end

function random_psd_matrix(
    rng::AbstractRNG, N::Int, lb::Float64, ub::Float64, ::Val{:dense}, ::SArrayStorage{T},
) where {T}
    return SMatrix{N, N, T}(random_psd_matrix(rng, N, lb, ub, Val(:dense), ArrayStorage(T)))
end

function random_psd_matrix(
    rng::AbstractRNG, N::Integer, lb::Float64, ub::Float64, storage::StorageType
)
    return random_psd_matrix(rng, N, lb, ub, Val(:dense), storage)
end

function random_psd_matrix(
    rng::AbstractRNG, N::Integer, lb::Float64, ub::Float64, ::Val{:diag}, ::ArrayStorage{T},
) where {T}
    return Diagonal(rand(rng, T, N) .* T(ub - lb) .+ T(lb))
end

function random_psd_matrix(
    rng::AbstractRNG, N::Integer, lb::Float64, ub::Float64, ::Val{:diag}, ::SArrayStorage{T},
) where {T}
    return Diagonal(rand(rng, SVector{N, T}) .* T(ub - lb) .+ T(lb))
end

function random_psd_matrix(rng::AbstractRNG, ::Integer, lb::Float64, ub::Float64, ::ScalarStorage{T}) where {T}
    return rand(rng, T) * T(ub - lb) + T(lb)
end



#
# Generation of Gaussians.
#

function random_gaussian(rng::AbstractRNG, dim::Int, s::StorageType)
    return Gaussian(random_vector(rng, dim, s), random_psd_matrix(rng, dim, 0.9, 1.1, s))
end

# Generation of SmallOutputLGC.

function random_small_output_lgc(
    rng::AbstractRNG, Dlat::Int, Dobs::Int, Q_type::Val, s::StorageType,
)
    return SmallOutputLGC(
        random_matrix(rng, Dobs, Dlat, s),
        random_vector(rng, Dobs, s),
        random_psd_matrix(rng, Dobs, 0.9, 1.1, Q_type, s),
    )
end

function random_scalar_output_lgc(rng::AbstractRNG, Dlat::Int, s::StorageType)
    return ScalarOutputLGC(
        random_vector(rng, Dlat, s)',
        randn(rng, eltype(s)),
        rand(rng, eltype(s)) + eltype(s)(1.0),
    )
end

function lgc_from_scalar_output_lgc(lgc::ScalarOutputLGC)
    return SmallOutputLGC(
        collect(lgc.A), [lgc.a], reshape([lgc.Q], 1, 1),
    )
end

function random_large_output_lgc(
    rng::AbstractRNG, Dlat::Int, Dobs::Int, Q_type::Val, s::StorageType,
)
    return LargeOutputLGC(
        random_matrix(rng, Dobs, Dlat, s),
        random_vector(rng, Dobs, s),
        random_psd_matrix(rng, Dobs, 0.9, 1.1, Q_type, s),
    )
end

function random_bottleneck_lgc(
    rng::AbstractRNG, Dlat::Int, Dmid::Int, Dobs::Int, Q_type::Val, s::StorageType,
)
    return BottleneckLGC(
        random_matrix(rng, Dmid, Dlat, s),
        random_vector(rng, Dmid, s),
        random_large_output_lgc(rng, Dmid, Dobs, Q_type, s),
    )
end

function small_output_lgc_from_bottleneck(model::BottleneckLGC)
    return SmallOutputLGC(
        model.fan_out.A * model.H,
        model.fan_out.a + model.fan_out.A * model.h,
        model.fan_out.Q,
    )
end


# Generation of GaussMarkovModels.

function random_tv_gmm(rng::AbstractRNG, ordering, Dlat::Int, N::Int, s::StorageType)

    As = map(_ -> random_matrix(rng, Dlat, Dlat, s), 1:N)
    as = map(_ -> random_vector(rng, Dlat, s), 1:N)
    x0 = random_gaussian(rng, Dlat, s)

    # For some reason this operation seems to be _incredibly_ inaccurate. My guess is that
    # the numerics are horrible for some reason, but I'm not sure why. Hence we add a pretty
    # large constant to the diagonal to ensure that all Qs are positive definite.
    Qs = map(n -> x0.P - Symmetric(As[n] * x0.P * As[n]') + eltype(s)(1.0) * I, 1:N)

    return GaussMarkovModel(ordering, As, as, Qs, x0)
end

function random_ti_gmm(rng::AbstractRNG, ordering, Dlat::Int, N::Int, s::StorageType)
    As = Fill(-random_psd_matrix(rng, Dlat, 0.1, 0.3, s), N)
    as = Fill(random_vector(rng, Dlat, s), N)
    x0 = random_gaussian(rng, Dlat, s)
    Qs = Fill(x0.P - As[1] * x0.P * As[1]', N)
    return GaussMarkovModel(ordering, As, as, Qs, x0)
end

# Generation of LGSSMs.

function random_lgssm(
    rng::AbstractRNG,
    ordering::Union{Forward, Reverse},
    ::Val{:time_varying},
    emission_type::Type{<:Union{SmallOutputLGC, LargeOutputLGC}},
    Dlat::Int,
    Dobs::Int,
    N::Int,
    Q_type::Val=Val{:dense},
    storage::StorageType=ArrayStorage(Float64),
)
    transitions = random_tv_gmm(rng, ordering, Dlat, N, storage)
    Hs = map(_ -> random_matrix(rng, Dobs, Dlat, storage), 1:N)
    hs = map(_ -> random_vector(rng, Dobs, storage), 1:N)
    Σs = map(_ -> random_psd_matrix(rng, Dobs, 0.9, 1.1, Q_type, storage), 1:N)
    T = emission_type{eltype(Hs), eltype(hs), eltype(Σs)}
    emissions = StructArray{T}((Hs, hs, Σs))
    return LGSSM(transitions, emissions)
end

function random_lgssm(
    rng::AbstractRNG,
    ordering::Union{Forward, Reverse},
    ::Val{:time_invariant},
    emission_type::Type{<:Union{SmallOutputLGC, LargeOutputLGC}},
    Dlat::Int,
    Dobs::Int,
    N::Int,
    Q_type::Val=Val{:dense},
    storage::StorageType=ArrayStorage(Float64),
)
    transitions = random_ti_gmm(rng, ordering, Dlat, N, storage)
    Hs = Fill(random_matrix(rng, Dobs, Dlat, storage), N)
    hs = Fill(random_vector(rng, Dobs, storage), N)
    Σs = Fill(random_psd_matrix(rng, Dobs, 0.9, 1.1, Q_type, storage), N)
    T = emission_type{eltype(Hs), eltype(hs), eltype(Σs)}
    emissions = StructArray{T}((Hs, hs, Σs))
    return LGSSM(transitions, emissions)
end

function random_lgssm(
    rng::AbstractRNG,
    ordering::Union{Forward, Reverse},
    ::Val{:time_varying},
    ::Type{ScalarOutputLGC},
    Dlat::Int,
    Dobs::Int,
    N::Int,
    storage::StorageType,
)
    transitions = random_tv_gmm(rng, ordering, Dlat, N, storage)
    Hs = map(_ -> random_vector(rng, Dlat, storage)', 1:N)
    hs = map(_ -> randn(rng, eltype(storage)), 1:N)
    Σs = map(_ -> convert(eltype(storage), rand(rng) + 0.1), 1:N)
    T = ScalarOutputLGC{eltype(Hs), eltype(hs), eltype(Σs)}
    emissions = StructArray{T}((Hs, hs, Σs))
    return LGSSM(transitions, emissions)
end

function random_lgssm(
    rng::AbstractRNG,
    ordering::Union{Forward, Reverse},
    ::Val{:time_invariant},
    ::Type{ScalarOutputLGC},
    Dlat::Int,
    Dobs::Int,
    N::Int,
    storage::StorageType,
)
    transitions = random_ti_gmm(rng, ordering, Dlat, N, storage)
    Hs = Fill(random_vector(rng, Dlat, storage)', N)
    hs = Fill(randn(rng, eltype(storage)), N)
    Σs = Fill(convert(eltype(storage), rand(rng) + 0.1), N)
    T = ScalarOutputLGC{eltype(Hs), eltype(hs), eltype(Σs)}
    emissions = StructArray{T}((Hs, hs, Σs))
    return LGSSM(transitions, emissions)
end

#
# Validation of internal consistency.
#

function validate_dims(model::Union{SmallOutputLGC, LargeOutputLGC})
    @test size(model.A) == (dim_out(model), dim_in(model))
    @test size(model.a) == (dim_out(model), )
    @test size(model.Q) == (dim_out(model), dim_out(model))
    return nothing
end

function validate_dims(model::ScalarOutputLGC)
    @test size(model.A) == (1, dim_in(model))
    @test size(model.a) == ()
    @test size(model.Q) == ()
    return nothing
end

function validate_dims(model::BottleneckLGC)
    validate_dims(model.fan_out)
    @test dim_in(model.fan_out) == size(model.H, 1)
    @test dim_in(model.fan_out) == length(model.h)
end

function validate_dims(gmm::GaussMarkovModel)

    # Check all vectors are the correct length.
    @test length(gmm.As) == length(gmm)
    @test length(gmm.as) == length(gmm)
    @test length(gmm.Qs) == length(gmm)

    # Check sizes of each element of the struct are correct.
    @test all(map(n -> dim_out(gmm[n]) == dim(gmm), eachindex(gmm)))
    @test all(map(n -> dim_in(gmm[n]) == dim(gmm), eachindex(gmm)))

    @test size(gmm.x0.m) == (dim(gmm),)
    @test size(gmm.x0.P) == (dim(gmm), dim(gmm))
    return nothing
end

function validate_dims(model::LGSSM)
    validate_dims(model.transitions)

    @test length(model) == length(model.transitions)
    @test length(model) == length(model.emissions)

    map(validate_dims, model.emissions)

    @test all(n -> dim(model.transitions) == dim_in(model.emissions[n]), eachindex(model))

    return nothing
end
