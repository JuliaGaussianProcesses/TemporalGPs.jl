"""
    ScalarLGSSM{Tmodel<:AbstractSSM} <: AbstractSSM

Linear Gaussian SSM whose outputs should be scalars. A lightweight wrapper around a regular
(vector-valued) LGSSM. Most of what this wrapper does is transform `AbstractVector`s of
`T <: Real`s into `AbstractVector`s of `SVector{1, T}`s, and then pass the data on to a
vector-valued ssm.
"""
struct ScalarLGSSM{Tmodel<:AbstractSSM} <: AbstractSSM
    model::Tmodel
end

Base.length(model::ScalarLGSSM) = length(model.model)
dim_obs(model::ScalarLGSSM) = 1
dim_latent(model::ScalarLGSSM) = dim_latent(model.model)

pick_first_scal(a::SVector{1, <:Real}, b) = first(a)
function get_pb(::typeof(pick_first_scal))
    pullback_pick_first_scal(Δ) = (SVector(Δ), nothing)
    pullback_pick_first_scal(::Nothing) = (nothing, nothing)
    return pullback_pick_first_scal
end

mean(model::ScalarLGSSM) = mean(model.model)
cov(model::ScalarLGSSM) = cov(model.model)

function correlate(model::ScalarLGSSM, αs::AbstractVector{<:Real}, f=pick_first_scal)
    αs_vec = reinterpret(SVector{1, eltype(αs)}, αs)
    lml, ys = correlate(model.model, αs_vec, f)
    return lml, ys
end

function decorrelate(model::ScalarLGSSM, ys::AbstractVector{<:Real}, f=pick_first_scal)
    ys_vec = reinterpret(SVector{1, eltype(ys)}, ys)
    lml, αs = decorrelate(model.model, ys_vec, f)
    return lml, αs
end

function whiten(model::ScalarLGSSM, ys::AbstractVector{<:Real})
    return last(decorrelate(model, ys))
end

function rand(rng::AbstractRNG, model::ScalarLGSSM)
    αs = randn(rng, length(model))
    return last(correlate(model, αs))
end

function unwhiten(model::ScalarLGSSM, αs::AbstractVector{<:Real})
    return last(correlate(model, αs))
end

function logpdf_and_rand(rng::AbstractRNG, model::ScalarLGSSM)
    αs = randn(rng, length(model))
    return correlate(model, αs)
end

function smooth(model::ScalarLGSSM, ys::AbstractVector{T}) where {T<:Real}
    return smooth(model.model, reinterpret(SVector{1, T}, ys))
end

function posterior_rand(rng::AbstractRNG, model::ScalarLGSSM, y::Vector{<:Real})
    fs = posterior_rand(rng, model.model, [SVector{1}(yn) for yn in y], 1)
    return first.(fs)
end
