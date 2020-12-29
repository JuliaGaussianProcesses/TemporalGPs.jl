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

Base.eachindex(model::ScalarLGSSM) = eachindex(model.model)

Base.getindex(model::ScalarLGSSM, t::Int) = ElementOfScalarSSM(model.model[t])

x0(model::ScalarLGSSM) = x0(model.model)

struct ElementOfScalarSSM{T}
    data::T
end

function step_marginals(x::Gaussian, model::ElementOfScalarSSM)
    y, x = step_marginals(x, model.data)
    return Gaussian(y.m[1], y.P[1]), x
end

function step_decorrelate(x::Gaussian, (model, y)::Tuple{ElementOfScalarSSM, Any})
    (lml, α_vec), x = step_decorrelate(x, (model.data, SVector{1}(y)))
    return (lml, α_vec[1]), x
end

function step_correlate(x::Gaussian, (model, α)::Tuple{ElementOfScalarSSM, Any})
    (lml, y_vec), x = step_correlate(x, (model.data, SVector{1}(α)))
    return (lml, y_vec[1]), x
end

rand_αs(rng::AbstractRNG, model::ScalarLGSSM) = randn(rng, length(model))


function get_adjoint_storage(x::ScalarLGSSM, Δx::NamedTuple{(:data,)})
    return (model=get_adjoint_storage(x.model, Δx.data),)
end

function _accum_at(Δxs::NamedTuple{(:model,)}, n::Int, Δx::NamedTuple{(:data,)})
    return (model = _accum_at(Δxs.model, n, Δx.data),)
end
