abstract type AbstractSSM end

"""
    LGSSM <: AbstractSSM

A linear-Gaussian state-space model. Represented in terms of a Gauss-Markov model `gmm` and
a vector of observation covariance matrices.
"""
struct LGSSM{Tgmm<:GaussMarkovModel, TΣ<:AV{<:AM{<:Real}}} <: AbstractSSM
    gmm::Tgmm
    Σ::TΣ
end

function Zygote._pullback(
    ::AContext, ::Type{<:LGSSM}, gmm::GaussMarkovModel, Σ::AV{<:AM{<:Real}},
)
    LGSSM_pullback(::Nothing) = (nothing, nothing, nothing)
    LGSSM_pullback(Δ::NamedTuple) = (nothing, Δ.gmm, Δ.Σ)
    return LGSSM(gmm, Σ), LGSSM_pullback
end

Base.:(==)(x::LGSSM, y::LGSSM) = (x.gmm == y.gmm) && (x.Σ == y.Σ)

Base.length(ft::LGSSM) = length(ft.gmm)

dim_obs(ft::LGSSM) = dim_obs(ft.gmm)

dim_latent(ft::LGSSM) = dim_latent(ft.gmm)

Base.eltype(ft::LGSSM) = eltype(ft.gmm)

storage_type(ft::LGSSM) = storage_type(ft.gmm)

Zygote.@nograd storage_type

function is_of_storage_type(ft::LGSSM, s::StorageType)
    return is_of_storage_type((ft.gmm, ft.Σ), s)
end

is_time_invariant(model::LGSSM) = false
is_time_invariant(model::LGSSM{<:GaussMarkovModel, <:Fill}) = is_time_invariant(model.gmm)

Base.getindex(model::LGSSM, n::Int) = (gmm = model.gmm[n], Σ = model.Σ[n])

mean(model::LGSSM) = mean(model.gmm)

function cov(model::LGSSM)
    S = Stheno.cov(model.gmm)
    Σ = Stheno.block_diagonal(model.Σ)
    return S + Σ
end

function predict(model::NamedTuple{(:gmm, :Σ)}, x)
    gmm = model.gmm
    return Gaussian(predict(x.m, x.P, gmm.A, gmm.a, gmm.Q)...)
end

function observe(model::NamedTuple{(:gmm, :Σ)}, x::Gaussian)
    return observe(model.gmm.H, model.gmm.h, model.Σ, x)
end

function observe(H::AbstractMatrix, h::AbstractVector, Σ::AbstractMatrix, x::Gaussian)
    return Gaussian(H * x.m + h, H * x.P * H' + Σ)
end

"""
    posterior_rand(rng::AbstractRNG, model::LGSSM, ys::Vector{<:AV{<:Real}})

Draw samples from the posterior over an LGSSM. This is not, currently, an especially
efficient implementation.
"""
function posterior_rand(
    rng::AbstractRNG,
    model::LGSSM,
    ys::Vector{<:AV{<:Real}},
    N_samples::Int,
)
    x_filter = _filter(model, ys)

    chol_Q = cholesky.(Symmetric.(model.gmm.Q .+ Ref(1e-15I)))

    x_T = rand(rng, x_filter[end], N_samples)
    x_sample = Vector{typeof(x_T)}(undef, length(ys))
    x_sample[end] = x_T
    for t in reverse(1:length(ys) - 1)

        # Produce joint samples.
        x̃ = rand(rng, x_filter[t], N_samples)
        x̃′ = model.gmm.A[t] * x̃ + model.gmm.a[t] + chol_Q[t].U' * randn(rng, size(x_T)...)

        # Applying conditioning transformation.
        AP = model.gmm.A[t] * x_filter[t].P
        S = Symmetric(model.gmm.A[t] * Matrix(transpose(AP)) + model.gmm.Q[t])
        chol_S = cholesky(S)

        x_sample[t] = x̃ + AP' * (chol_S.U \ (chol_S.U' \ (x_sample[t+1] - x̃′)))
    end

    return map(n -> model.gmm.H[n] * x_sample[n] .+ model.gmm.h[n], eachindex(x_sample))
end

function posterior_rand(rng::AbstractRNG, model::LGSSM, y::Vector{<:Real}, N_samples::Int)
    return posterior_rand(rng, model, [SVector{1}(yn) for yn in y], N_samples)
end



#
# Things defined in terms of decorrelate
#

Stheno.logpdf(model::AbstractSSM, ys::AbstractVector) = decorrelate(model, ys)[1]

whiten(model::AbstractSSM, ys::AbstractVector) = decorrelate(model, ys)[2]

_filter(model::AbstractSSM, ys::AbstractVector) = decorrelate(model, ys)[3]


#
# Things defined in terms of correlate
#

function Random.rand(rng::AbstractRNG, model::AbstractSSM)
    return correlate(model, rand_αs(rng, model))[2] # last isn't type-stable inside AD.
end

unwhiten(model::AbstractSSM, αs::AbstractVector) = correlate(model, αs)[2]

function logpdf_and_rand(rng::AbstractRNG, model::AbstractSSM)
    lml, ys, _ = correlate(model, rand_αs(rng, model))
    return lml, ys
end

function rand_αs(rng::AbstractRNG, model::LGSSM)
    D = dim_obs(model)
    α = randn(rng, eltype(model), length(model) * D)
    return [α[(n - 1) * D + 1:n * D] for n in 1:length(model)]
end

function rand_αs(rng::AbstractRNG, model::LGSSM{<:GaussMarkovModel{<:AV{<:SArray}}})
    D = dim_obs(model)
    ot = output_type(model)
    α = randn(rng, eltype(model), length(model) * D)

    # For some type-stability reasons, we have to ensure that
    αs = Vector{output_type(model)}(undef, length(model))
    map(n -> setindex!(αs, ot(α[(n - 1) * D + 1:n * D]), n), 1:length(model))
    return αs
end

ChainRulesCore.@non_differentiable rand_αs(::AbstractRNG, ::AbstractSSM)

output_type(ft::LGSSM{<:GaussMarkovModel{<:AV{<:SArray}}}) = eltype(ft.gmm.h)
