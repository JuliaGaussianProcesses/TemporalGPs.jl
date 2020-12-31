"""
    LGSSM <: AbstractLGSSM

A linear-Gaussian state-space model. Represented in terms of a Gauss-Markov model `gmm` and
a vector of observation covariance matrices.
"""
struct LGSSM{
    Tgmm <: GaussMarkovModel,
    THs <: AbstractVector,
    Ths <: AbstractVector,
    TΣs <: AbstractVector,
} <: AbstractLGSSM
    gmm::Tgmm
    Hs::THs
    hs::Ths
    Σs::TΣs
end

ordering(model::LGSSM) = ordering(model.gmm)

function Base.:(==)(x::LGSSM, y::LGSSM)
    return (x.gmm == y.gmm) && (x.Hs == y.Hs) && (x.hs == y.hs) && (x.Σs == y.Σs)
end

Base.length(model::LGSSM) = length(model.gmm)

Base.eachindex(model::LGSSM) = eachindex(model.gmm)

dim_obs(model::LGSSM) = length(first(model.hs))

dim_latent(model::LGSSM) = dim(model.gmm)

Base.eltype(model::LGSSM) = eltype(model.gmm)

storage_type(model::LGSSM) = storage_type(model.gmm)

Zygote.@nograd storage_type

function is_of_storage_type(model::LGSSM, s::StorageType)
    return is_of_storage_type((model.gmm, model.Hs, model.hs, model.Σs), s)
end

x0(model::LGSSM) = x0(model.gmm)

function rand_αs(rng::AbstractRNG, model::LGSSM{<:GaussMarkovModel{<:AV{<:SArray}}})
    return map(_ -> randn(rng, SVector{dim_obs(model), eltype(model)}), 1:length(model))
end

struct ElementOfLGSSM{Tordering, Ttransition, Temission}
    ordering::Tordering
    transition::Ttransition
    emission::Temission
end

ordering(x::ElementOfLGSSM) = x.ordering

transition_dynamics(x::ElementOfLGSSM) = x.transition

emission_dynamics(x::ElementOfLGSSM) = x.emission

function Base.getindex(model::LGSSM, n::Int)
    return ElementOfLGSSM(
        ordering(model),
        model.gmm[n],
        LinearGaussianDynamics(model.Hs[n], model.hs[n], model.Σs[n]),
    )
end


# AD stuff. No need to understand this.

function get_adjoint_storage(x::LGSSM, Δx::NamedTuple{(:ordering, :transition, :emission)})
    return (
        gmm = get_adjoint_storage(x.gmm, Δx.transition),
        Hs = get_adjoint_storage(x.Hs, Δx.emission.A),
        hs = get_adjoint_storage(x.hs, Δx.emission.a),
        Σs = get_adjoint_storage(x.Σs, Δx.emission.Q),
    )
end

function _accum_at(
    Δxs::NamedTuple{(:gmm, :Hs, :hs, :Σs)},
    n::Int,
    Δx::NamedTuple{(:ordering, :transition, :emission)},
)
    return (
        gmm = _accum_at(Δxs.gmm, n, Δx.transition),
        Hs = _accum_at(Δxs.Hs, n, Δx.emission.A),
        hs = _accum_at(Δxs.hs, n, Δx.emission.a),
        Σs = _accum_at(Δxs.Σs, n, Δx.emission.Q),
    )
end
