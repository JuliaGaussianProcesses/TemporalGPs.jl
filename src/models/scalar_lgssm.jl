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

is_of_storage_type(model::ScalarLGSSM, s::StorageType) = is_of_storage_type(model.model, s)

is_time_invariant(model::ScalarLGSSM) = is_time_invariant(model.model)

# Converts a vector of observations to a vector of 1-vectors.
to_vector_observations(::ArrayStorage{T}, y::AV{T}) where {T<:Real} = [[y_] for y_ in y]

@adjoint function to_vector_observations(s::StorageType{T}, y::AV{T}) where {T<:Real}
    function pullback_to_vector_observations(Δ::AV)
        return (nothing, from_vector_observations(Δ))
    end
    return to_vector_observations(s, y), pullback_to_vector_observations
end

function to_vector_observations(::SArrayStorage{T}, y::AV{T}) where {T<:Real}
    return map(SVector{1, eltype(y)}, y)
end

# Converts a vector of 1-vectors into a vector of reals.
from_vector_observations(ys::AV{<:AV{<:Real}}) = map(first, ys)

@adjoint function from_vector_observations(ys::AV{<:AV{<:Real}})
    function pullback_from_vector_observations(Δ::AbstractVector{<:Real})
        return (map(x -> [x], Δ), )
    end
    return from_vector_observations(ys), pullback_from_vector_observations
end

@adjoint function from_vector_observations(ys::AV{<:SVector{1, <:Real}})
    function pullback_from_vector_observations(Δ::AbstractVector{<:Real})
        return (map(SVector{1, eltype(Δ)}, Δ),)
    end
    return from_vector_observations(ys), pullback_from_vector_observations
end

# Converts a vector of 1-dimensional Gaussians into a vector of Normals.
function from_vector_observations(ys::AV{<:Gaussian})
    return Normal.(only.(getfield.(ys, :m)), sqrt.(only.(getfield.(ys, :P))))
end

function Stheno.marginals(model::ScalarLGSSM)
    return from_vector_observations(Stheno.marginals(model.model))
end

function decorrelate(model::ScalarLGSSM, ys::AbstractVector{<:Real}, f::typeof(copy_first))
    ys_vec = to_vector_observations(storage_type(model), ys)
    lml, αs = decorrelate(model.model, ys_vec, f)
    return lml, from_vector_observations(αs)
end

function decorrelate(model::ScalarLGSSM, ys::AbstractVector{<:Real}, f::typeof(pick_last))
    ys_vec = to_vector_observations(storage_type(model), ys)
    return decorrelate(model.model, ys_vec, f)
end

function correlate(model::ScalarLGSSM, αs::AbstractVector{<:Real}, f::typeof(copy_first))
    αs_vec = to_vector_observations(storage_type(model), αs)
    lml, ys = correlate(model.model, αs_vec, f)
    return lml, from_vector_observations(ys)
end

function correlate(model::ScalarLGSSM, αs::AbstractVector{<:Real}, f::typeof(pick_last))
    αs_vec = to_vector_observations(storage_type(model), αs)
    return correlate(model.model, αs_vec, f)
end

rand_αs(rng::AbstractRNG, model::ScalarLGSSM) = randn(rng, length(model))

function smooth(model::ScalarLGSSM, ys::AbstractVector{T}) where {T<:Real}
    return smooth(model.model, to_vector_observations(storage_type(model), ys))
end

function posterior_rand(rng::AbstractRNG, model::ScalarLGSSM, y::Vector{<:Real})
    fs = posterior_rand(rng, model.model, [SVector{1}(yn) for yn in y], 1)
    return first.(fs)
end

checkpointed(model::ScalarLGSSM) = ScalarLGSSM(checkpointed(model.model))
