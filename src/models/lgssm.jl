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

Base.:(==)(x::LGSSM, y::LGSSM) = (x.gmm == y.gmm) && (x.Σ == y.Σ)

Base.length(ft::LGSSM) = length(ft.gmm)

Base.eachindex(model::LGSSM) = 1:length(model)

dim_obs(ft::LGSSM) = dim_obs(ft.gmm)

dim_latent(ft::LGSSM) = dim_latent(ft.gmm)

Base.eltype(ft::LGSSM) = eltype(ft.gmm)

storage_type(ft::LGSSM) = storage_type(ft.gmm)

Zygote.@nograd storage_type

is_of_storage_type(ft::LGSSM, s::StorageType) = is_of_storage_type((ft.gmm, ft.Σ), s)

x0(model::LGSSM) = x0(model.gmm)

is_time_invariant(model::LGSSM) = false

is_time_invariant(model::LGSSM{<:GaussMarkovModel, <:Fill}) = is_time_invariant(model.gmm)

Base.getindex(model::LGSSM, n::Int) = (gmm = model.gmm[n], Σ = model.Σ[n])

mean(model::LGSSM) = mean(model.gmm)

function cov(model::LGSSM)
    S = Stheno.cov(model.gmm)
    Σ = Stheno.block_diagonal(model.Σ)
    return S + Σ
end

function Stheno.marginals(model::AbstractSSM)
    return scan_emit(step_marginals, model, x0(model), eachindex(model))[1]
end

function Stheno.logpdf(model::AbstractSSM, y::AbstractVector)
    pick_lml(((lml, _), x)) = (lml, x)
    return sum(scan_emit(
        pick_lml ∘ step_decorrelate, zip(model, y), x0(model), eachindex(model),
    )[1])
end

function decorrelate(model::AbstractSSM, y::AbstractVector)
    pick_α(((_, α), x)) = (α, x)
    α, _ = scan_emit(pick_α ∘ step_decorrelate, zip(model, y), x0(model), eachindex(model))
    return α
end

function _filter(model::AbstractSSM, y::AbstractVector)
    pick_x((_, x)) = (x, x)
    xs, _ = scan_emit(pick_x ∘ step_decorrelate, zip(model, y), x0(model), eachindex(model))
    return xs
end

function correlate(model::AbstractSSM, α::AbstractVector)
    pick_y(((_, y), x)) = (y, x)
    ys, _ = scan_emit(pick_y ∘ step_correlate, zip(model, α), x0(model), eachindex(model))
    return ys
end

Stheno.rand(rng::AbstractRNG, model::AbstractSSM) = correlate(model, rand_αs(rng, model))

function rand_αs(rng::AbstractRNG, model::AbstractSSM)
    return map(_ -> randn(rng, eltype(model), dim_obs(model)), 1:length(model))
end

function rand_αs(rng::AbstractRNG, model::LGSSM{<:GaussMarkovModel{<:AV{<:SArray}}})
    return map(_ -> randn(rng, SVector{dim_obs(model), eltype(model)}), 1:length(model))
end

ChainRulesCore.@non_differentiable rand_αs(::AbstractRNG, ::AbstractSSM)

#
# step
#

function step_marginals(x::Gaussian, model::NamedTuple{(:gmm, :Σ)})
    x = predict(model, x)
    y = observe(model, x)
    return y, x
end

function step_decorrelate(x::Gaussian, (model, y)::Tuple{NamedTuple{(:gmm, :Σ)}, Any})
    gmm = model.gmm
    mp, Pp = predict(x.m, x.P, gmm.A, gmm.a, gmm.Q)
    mf, Pf, lml, α = update_decorrelate(mp, Pp, gmm.H, gmm.h, model.Σ, y)
    return (lml, α), Gaussian(mf, Pf)
end

function step_correlate(x::Gaussian, (model, α)::Tuple{NamedTuple{(:gmm, :Σ)}, Any})
    gmm = model.gmm
    mp, Pp = predict(x.m, x.P, gmm.A, gmm.a, gmm.Q)
    mf, Pf, lml, y = update_correlate(mp, Pp, gmm.H, gmm.h, model.Σ, α)
    return (lml, y), Gaussian(mf, Pf)
end



#
# predict and update
#

function predict(mf::AV{T}, Pf::AM{T}, A::AM{T}, a::AV{T}, Q::AM{T}) where {T<:Real}
    return A * mf + a, (A * Pf) * A' + Q
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

function update_decorrelate(
    mp::AV{T}, Pp::AM{T}, H::AM{T}, h::AV{T}, Σ::AM{T}, y::AV{T},
) where {T<:Real}
    V = H * Pp
    S_1 = V * H' + Σ
    S = cholesky(Symmetric(S_1))
    U = S.U
    B = U' \ V
    α = U' \ (y - H * mp - h)

    mf = mp + B'α
    Pf = _compute_Pf(Pp, B)
    lml = -(length(y) * T(log(2π)) + logdet(S) + α'α) / 2
    return mf, Pf, lml, α
end

function update_correlate(
    mp::AV{T}, Pp::AM{T}, H::AM{T}, h::AV{T}, Σ::AM{T}, α::AV{T},
) where {T<:Real}

    V = H * Pp
    S = cholesky(Symmetric(V * H' + Σ))
    B = S.U' \ V
    y = S.U'α + (H * mp + h)

    mf = mp + B'α
    Pf = _compute_Pf(Pp, B)
    lml = -(length(y) * T(log(2π)) + logdet(S) + α'α) / 2
    return mf, Pf, lml, y
end

_compute_Pf(Pp::AM{T}, B::AM{T}) where {T<:Real} = Pp - B'B

# function _compute_Pf(Pp::Matrix{T}, B::Matrix{T}) where {T<:Real}
#     # Copy of Pp is necessary to ensure that the memory isn't modified.
#     # return BLAS.syrk!('U', 'T', -one(T), B, one(T), copy(Pp))
#     # I probably _do_ need a custom adjoint for this...
#     return LinearAlgebra.copytri!(BLAS.syrk!('U', 'T', -one(T), B, one(T), copy(Pp)), 'U')
# end

function get_adjoint_storage(x::LGSSM, Δx::NamedTuple{(:gmm, :Σ)})
    return (gmm = get_adjoint_storage(x.gmm, Δx.gmm), Σ = get_adjoint_storage(x.Σ, Δx.Σ))
end

function _accum_at(Δxs::NamedTuple{(:gmm, :Σ)}, n::Int, Δx::NamedTuple{(:gmm, :Σ)})
    return (gmm = _accum_at(Δxs.gmm, n, Δx.gmm), Σ = _accum_at(Δxs.Σ, n, Δx.Σ))
end

function Zygote._pullback(
    ::AContext, ::Type{<:LGSSM}, gmm::GaussMarkovModel, Σ::AV{<:AM{<:Real}},
)
    LGSSM_pullback(::Nothing) = (nothing, nothing, nothing)
    LGSSM_pullback(Δ::NamedTuple) = (nothing, Δ.gmm, Δ.Σ)
    return LGSSM(gmm, Σ), LGSSM_pullback
end
