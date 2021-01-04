abstract type AbstractLGSSM end

"""
    LGSSM{Ttransitions<:GaussMarkovModel, Temissions<:StructArray} <: AbstractLGSSM

A linear-Gaussian state-space model. Represented in terms of a Gauss-Markov model
`transitions` and collection of emission dynamics `emissions`.
"""
struct LGSSM{Ttransitions<:GaussMarkovModel, Temissions<:StructArray} <: AbstractLGSSM
    transitions::Ttransitions
    emissions::Temissions
end

ordering(model::LGSSM) = ordering(model.transitions)

function Base.:(==)(x::LGSSM, y::LGSSM)
    return (x.transitions == y.transitions) && (x.emissions == y.emissions)
end

Base.length(model::LGSSM) = length(model.transitions)

Base.eachindex(model::LGSSM) = eachindex(model.transitions)

storage_type(model::LGSSM) = storage_type(model.transitions)

Zygote.@nograd storage_type

function is_of_storage_type(model::LGSSM, s::StorageType)
    return is_of_storage_type((model.transitions, model.emissions), s)
end

# Get the (Gaussian) distribution over the latent process at time t=0.
x0(model::LGSSM) = x0(model.transitions)



# Functionality for indexing into an LGSSM.

struct ElementOfLGSSM{Tordering, Ttransition, Temission}
    ordering::Tordering
    transition::Ttransition
    emission::Temission
end

ordering(x::ElementOfLGSSM) = x.ordering

transition_dynamics(x::ElementOfLGSSM) = x.transition

emission_dynamics(x::ElementOfLGSSM) = x.emission

function Base.getindex(model::LGSSM, n::Int)
    return ElementOfLGSSM(ordering(model), model.transitions[n], model.emissions[n])
end



# Draw a sample from the model.

function Stheno.rand(rng::AbstractRNG, model::LGSSM)
    iterable = zip(ε_randn(rng, model), model)
    init = rand(rng, x0(model))
    return scan_emit(step_rand, iterable, init, eachindex(model))[1]
end

# Generate randomness used only once so that checkpointing works.
function ε_randn(rng::AbstractRNG, model::LGSSM)
    return zip(
        map(n -> ε_randn(rng, model.transitions[n]), eachindex(model.transitions)),
        map(f -> ε_randn(rng, f), model.emissions),
    )
end

step_rand(x::AbstractVector, (rng, model)) = step_rand(ordering(model), x, (rng, model))

function step_rand(::Forward, x::AbstractVector, ((ε_t, ε_e), model))
    x_next = conditional_rand(ε_t, transition_dynamics(model), x)
    y = conditional_rand(ε_e, emission_dynamics(model), x_next)
    return y, x_next
end

function step_rand(::Reverse, x::AbstractVector, ((ε_t, ε_e), model))
    y = conditional_rand(ε_e, emission_dynamics(model), x)
    x_next = conditional_rand(ε_t, transition_dynamics(model), x)
    return y, x_next
end



# Compute the time-marginals of the output of the model.

function Stheno.marginals(model::LGSSM)
    return scan_emit(step_marginals, model, x0(model), eachindex(model))[1]
end

step_marginals(x::Gaussian, model) = step_marginals(ordering(model), x, model)

function step_marginals(::Forward, x::Gaussian, model)
    xp = predict(x, transition_dynamics(model))
    y = predict(xp, emission_dynamics(model))
    return y, xp
end

function step_marginals(::Reverse, x::Gaussian, model)
    y = predict(x, emission_dynamics(model))
    xp = predict(x, transition_dynamics(model))
    return y, xp
end



# Compute the log marginal likelihood of the observations `y`.

function Stheno.logpdf(model::LGSSM, y::AbstractVector)
    return sum(scan_emit(step_logpdf, zip(model, y), x0(model), eachindex(model))[1])
end

step_logpdf(x::Gaussian, (model, y)) = step_logpdf(ordering(model), x, (model, y))

function step_logpdf(::Forward, x::Gaussian, (model, y))
    xp = predict(x, transition_dynamics(model))
    xf, lml = posterior_and_lml(xp, emission_dynamics(model), y)
    return lml, xf
end

function step_logpdf(::Reverse, x::Gaussian, (model, y))
    xf, lml = posterior_and_lml(x, emission_dynamics(model), y)
    xp = predict(xf, transition_dynamics(model))
    return lml, xp
end



# Compute the filtering distributions.

function _filter(model::LGSSM, y::AbstractVector)
    return scan_emit(step_filter, zip(model, y), x0(model), eachindex(model))[1]
end

step_filter(x::Gaussian, (model, y)) = step_filter(ordering(model), x, (model, y))

function step_filter(::Forward, x::Gaussian, (model, y))
    xp = predict(x, transition_dynamics(model))
    xf, lml = posterior_and_lml(xp, emission_dynamics(model), y)
    return xf, xf
end

function step_filter(::Reverse, x::Gaussian, (model, y))
    xf, lml, α = decorrelate(x, emission_dynamics(model), y)
    xp = predict(xf, transition_dynamics(model))
    return xf, xp
end



# Construct the posterior model.

function posterior(prior::LGSSM, y::AbstractVector)
    new_trans, xf = scan_emit(step_posterior, zip(prior, y), x0(prior), eachindex(prior))
    A = map(x -> x.A, new_trans)
    a = map(x -> x.a, new_trans)
    Q = map(x -> x.Q, new_trans)
    return LGSSM(GaussMarkovModel(Reverse(), A, a, Q, xf), prior.emissions)
end

step_posterior(xf::Gaussian, (prior, y)) = step_posterior(ordering(prior), xf, (prior, y))

function step_posterior(::Forward, xf::Gaussian, (prior, y))
    t = transition_dynamics(prior)
    xp = predict(xf, t)
    xf, _ = posterior_and_lml(xp, emission_dynamics(prior), y)
    return invert_dynamics(xf, xp, t), xf
end

# inlining for the benefit of type inference. Needed in at least julia-1.5.3.
@inline function invert_dynamics(xf::Gaussian, xp::Gaussian, prior::SmallOutputLGC)
    U = cholesky(Symmetric(xp.P + ident_eps(xf))).U
    Gt = U \ (U' \ (prior.A * xf.P))
    return SmallOutputLGC(_collect(Gt'), xf.m - Gt'xp.m, _compute_Pf(xf.P, U * Gt))
end

_compute_Pf(Pp::AbstractMatrix, B::AbstractMatrix) = Pp - B'B

ident_eps(xf) = UniformScaling(convert(eltype(xf), 1e-12))

function Zygote._pullback(::NoContext, ::typeof(ident_eps), xf)
    ident_eps_pullback(Δ) = nothing
    return ident_eps(xf), ident_eps_pullback
end

_collect(U::Adjoint{<:Any, <:Matrix}) = collect(U)
_collect(U::SMatrix) = U

small_noise_cov(::Type{<:SMatrix{D, D, T}}, ::Int) where {D, T} = SMatrix{D, D, T}(1e-12I)

small_noise_cov(::Type{Matrix{T}}, D::Int) where {T} = Matrix{T}(1e-12 * I, D, D)



# AD stuff. No need to understand this unless you're really plumbing the depths...

function get_adjoint_storage(x::LGSSM, Δx::NamedTuple{(:ordering, :transition, :emission)})
    return (
        transitions = get_adjoint_storage(x.transitions, Δx.transition),
        emissions = get_adjoint_storage(x.emissions, Δx.emission)
    )
end

function _accum_at(
    Δxs::NamedTuple{(:transitions, :emissions)},
    n::Int,
    Δx::NamedTuple{(:ordering, :transition, :emission)},
)
    return (
        transitions = _accum_at(Δxs.transitions, n, Δx.transition),
        emissions = _accum_at(Δxs.emissions, n, Δx.emission),
    )
end
