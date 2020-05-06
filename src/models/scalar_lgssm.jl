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

Base.eltype(model::ScalarLGSSM) = eltype(model.model)

dim_obs(model::ScalarLGSSM) = 1

dim_latent(model::ScalarLGSSM) = dim_latent(model.model)

storage_type(model::ScalarLGSSM) = storage_type(model.model)

mean(model::ScalarLGSSM) = mean(model.model)

cov(model::ScalarLGSSM) = cov(model.model)

# Converts a vector of observations to a vector of 1-vectors.
to_vector_observations(::ArrayStorage{T}, y::AV{T}) where {T<:Real} = [[y_] for y_ in y]

function to_vector_observations(::SArrayStorage{T}, y::AV{T}) where {T<:Real}
    return reinterpret(SVector{1, eltype(y)}, y)
end

# Converts a vector of 1-vectors into a vector of reals.
from_vector_observations(ys::AV{<:AV{T}}) where {T<:Real} = first.(ys)

@adjoint function from_vector_observations(ys::AV{<:SVector{1, T}}) where {T<:Real}
    function pullback_from_vector_observations(Δ::AbstractVector{<:Real})
        return (SVector{1, eltype(Δ)}.(Δ),)
    end
    return from_vector_observations(ys), pullback_from_vector_observations
end

function correlate(model::ScalarLGSSM, αs::AbstractVector{<:Real}, f=copy_first)
    storage = storage_type(model)
    αs_vec = to_vector_observations(storage, αs)
    lml, ys = correlate(model.model, αs_vec, f)
    return lml, from_vector_observations(ys)
end

function decorrelate(model::ScalarLGSSM, ys::AbstractVector{<:Real}, f=copy_first)
    return decorrelate(mutability(storage_type(model)), model, ys, f)
end

function decorrelate(mut, model::ScalarLGSSM, ys::AbstractVector{<:Real}, f=copy_first)
    storage = storage_type(model)
    ys_vec = to_vector_observations(storage, ys)
    lml, αs = decorrelate(mut, model.model, ys_vec, f)
    return lml, from_vector_observations(αs)
end


function whiten(model::ScalarLGSSM, ys::AbstractVector{<:Real})
    return last(decorrelate(model, ys))
end

function rand(rng::AbstractRNG, model::ScalarLGSSM)
    αs = randn(rng, eltype(model), length(model))
    return last(correlate(model, αs))
end

function unwhiten(model::ScalarLGSSM, αs::AbstractVector{<:Real})
    return last(correlate(model, αs))
end

function logpdf_and_rand(rng::AbstractRNG, model::ScalarLGSSM)
    αs = randn(rng, eltype(model), length(model))
    return correlate(model, αs)
end

function smooth(model::ScalarLGSSM, ys::AbstractVector{T}) where {T<:Real}
    return smooth(model.model, reinterpret(SVector{1, T}, ys))
end

function posterior_rand(rng::AbstractRNG, model::ScalarLGSSM, y::Vector{<:Real})
    fs = posterior_rand(rng, model.model, [SVector{1}(yn) for yn in y], 1)
    return first.(fs)
end
