using TemporalGPs:
    GaussMarkovModel,
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
    return randn(rng, T, M, N)
end

function random_matrix(rng::AbstractRNG, M::Int, N::Int, ::SArrayStorage{T}) where {T<:Real}
    return SMatrix{M, N}(randn(rng, T, M, N))
end

function random_nice_psd_matrix(
    rng::AbstractRNG, N::Integer, ::Val{:dense}, ::ArrayStorage{T},
) where {T}

    # Generate random positive definite matrix.
    S = Symmetric(pw(Matern12(), 5 .* randn(rng, T, N)) + T(1e-3) * I)

    # Centre (make eigenvals N(0, 2^2)) and bound the eigenvalues between 0 and 1.
    λ, Γ = eigen(S)
    m_λ = N > 1 ? mean(λ) : mean(λ) + T(0.1)
    σ_λ = N > 1 ? std(λ) : T(1.0)
    λ .= 2 .* (λ .- (m_λ + 0.1)) ./ σ_λ
    @. λ = T(1 / (1 + exp(-λ)) * 0.9 + 0.1)
    return collect(Symmetric(Γ * Diagonal(λ) * Γ'))
end

function random_nice_psd_matrix(
    rng::AbstractRNG, N::Integer, ::Val{:dense}, ::SArrayStorage{T},
) where {T}
    return SMatrix{N, N, T}(random_nice_psd_matrix(rng, N, Val(:dense), ArrayStorage(T)))
end

function random_nice_psd_matrix(rng::AbstractRNG, N::Integer, storage::StorageType)
    return random_nice_psd_matrix(rng, N, Val(:dense), storage)
end

function random_nice_psd_matrix(
    rng::AbstractRNG, N::Integer, ::Val{:diag}, ::ArrayStorage{T},
) where {T}
    return Diagonal(rand(T, N) .+ T(0.1))
end

function random_nice_psd_matrix(
    rng::AbstractRNG, N::Integer, ::Val{:diag}, ::SArrayStorage{T},
) where {T}
    return Diagonal(rand(SVector{N, T}) .+ T(0.1))
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



# Generation of SmallOutputLGC.

function random_small_output_lgc(
    rng::AbstractRNG, Dlat::Int, Dobs::Int, Q_type::Val, s::StorageType,
)
    return SmallOutputLGC(
        random_matrix(rng, Dobs, Dlat, s),
        random_vector(rng, Dobs, s),
        random_nice_psd_matrix(rng, Dobs, Q_type, s),
    )
end

function random_scalar_output_lgc(rng::AbstractRNG, Dlat::Int, s::StorageType)
    return ScalarOutputLGC(
        random_vector(rng, Dlat, s)',
        randn(rng, eltype(s)),
        rand(rng, eltype(s)) + 0.1,
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
        random_nice_psd_matrix(rng, Dobs, Q_type, s),
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

    As = map(_ -> -random_nice_psd_matrix(rng, Dlat, s), 1:N)
    as = map(_ -> random_vector(rng, Dlat, s), 1:N)
    x0 = random_gaussian(rng, Dlat, s)

    # For some reason this operation seems to be _incredibly_ inaccurate. My guess is that
    # the numerics are horrible for some reason, but I'm not sure why. Hence we add a pretty
    # large constant to the diagonal to ensure that all Qs are positive definite.
    Qs = map(n -> x0.P - Symmetric(As[n] * x0.P * As[n]') + eltype(s)(1e-1) * I, 1:N)

    return GaussMarkovModel(ordering, As, as, Qs, x0)
end

function random_ti_gmm(rng::AbstractRNG, ordering, Dlat::Int, N::Int, s::StorageType)

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
        Qs = gmm_Qs_tangent(rng, gmm.Qs, storage_type(gmm)),
        x0 = rand_tangent(rng, gmm.x0),
    )
end

function gmm_Qs_tangent(
    rng::AbstractRNG, Qs::T, storage_type::StorageType,
) where {T<:Vector{<:AbstractMatrix}}
    return map(Q -> random_nice_psd_matrix(rng, size(Q, 1), storage_type), Qs)
end

function gmm_Qs_tangent(
    rng::AbstractRNG, Qs::T, storage_type::StorageType,
) where {T<:Fill{<:AbstractMatrix}}
    Δ = random_nice_psd_matrix(rng, size(first(Qs), 1), storage_type)
    return Composite{T}(value=Δ)
end

function gmm_Qs_tangent(
    rng::AbstractRNG, Qs::T, storage_type::StorageType,
) where {T<:Vector{<:Real}}
    return map(Q -> convert(eltype(storage_type), rand(rng) + 0.1), Qs)
end

function gmm_Qs_tangent(
    rng::AbstractRNG, Qs::T, storage_type::StorageType,
) where {T<:Fill{<:Real}}
    return Composite{T}(value=convert(eltype(storage_type), rand(rng) + 0.1))
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
    Σs = map(_ -> random_nice_psd_matrix(rng, Dobs, Q_type, storage), 1:N)
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
    Σs = Fill(random_nice_psd_matrix(rng, Dobs, Q_type, storage), N)
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

function FiniteDifferences.rand_tangent(rng::AbstractRNG, ssm::T) where {T<:LGSSM}
    Hs = ssm.emissions.A
    hs = ssm.emissions.a
    Σs = ssm.emissions.Q
    return Composite{T}(
        transitions = rand_tangent(rng, ssm.transitions),
        emissions = Composite{typeof(ssm.emissions)}(fieldarrays=(
            A=rand_tangent(rng, Hs),
            a=rand_tangent(rng, hs),
            Q=gmm_Qs_tangent(rng, Σs, storage_type(ssm)),
        )),
    )
end

# function random_tv_scalar_lgssm(rng::AbstractRNG, Dlat::Int, N::Int, storage)
#     return ScalarLGSSM(random_tv_lgssm(rng, Dlat, 1, N, storage))
# end

# function random_ti_scalar_lgssm(rng::AbstractRNG, Dlat::Int, N::Int, storage)
#     return ScalarLGSSM(random_ti_lgssm(rng, Dlat, 1, N, storage))
# end

# function random_tv_posterior_lgssm(rng::AbstractRNG, Dlat::Int, Dobs::Int, N::Int, storage)
#     lgssm = random_tv_lgssm(rng, Dlat, Dobs, N, storage)
#     y = rand(rng, lgssm)
#     Σs = map(_ -> random_nice_psd_matrix(rng, Dobs, storage), eachindex(y))
#     return posterior(lgssm, y, Σs)
# end

# function random_ti_posterior_lgssm(rng::AbstractRNG, Dlat::Int, Dobs::Int, N::Int, storage)
#     lgssm = random_ti_lgssm(rng, Dlat, Dobs, N, storage)
#     y = rand(rng, lgssm)
#     Σs = Fill(random_nice_psd_matrix(rng, Dobs, storage), length(lgssm))
#     return posterior(lgssm, y, Σs)
# end


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

# function __verify_model_properties(model, Dlat, Dobs, N, storage_type)
#     @test is_of_storage_type(model, storage_type)
#     @test length(model) == N
#     @test dim_obs(model) == Dobs
#     @test dim_latent(model) == Dlat
#     validate_dims(model)
#     return nothing
# end

# function __verify_model_properties(model, Dlat, N, storage_type)
#     return __verify_model_properties(model, Dlat, 1, N, storage_type)
# end
