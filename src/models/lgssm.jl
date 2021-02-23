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

@inline function transitions(model::LGSSM)
    return Zygote.literal_getfield(model, Val(:transitions))
end

@inline function emissions(model::LGSSM)
    return Zygote.literal_getfield(model, Val(:emissions))
end

@inline ordering(model::LGSSM) = ordering(transitions(model))

Zygote._pullback(::AContext, ::typeof(ordering), model) = ordering(model), nograd_pullback

function Base.:(==)(x::LGSSM, y::LGSSM)
    return (transitions(x) == transitions(y)) && (emissions(x) == emissions(y))
end

Base.length(model::LGSSM) = length(transitions(model))

Base.eachindex(model::LGSSM) = eachindex(transitions(model))

storage_type(model::LGSSM) = storage_type(transitions(model))

Zygote.@nograd storage_type

function is_of_storage_type(model::LGSSM, s::StorageType)
    return is_of_storage_type((transitions(model), emissions(model)), s)
end

# Get the (Gaussian) distribution over the latent process at time t=0.
x0(model::LGSSM) = x0(transitions(model))

emission_type(model::LGSSM) = eltype(emissions(model))



# Functionality for indexing into an LGSSM.

struct ElementOfLGSSM{Tordering, Ttransition, Temission}
    ordering::Tordering
    transition::Ttransition
    emission::Temission
end

@inline function ordering(x::ElementOfLGSSM)
    return Zygote.literal_getfield(x, Val(:ordering))
end

@inline function transition_dynamics(x::ElementOfLGSSM)
    return Zygote.literal_getfield(x, Val(:transition))
end

@inline function emission_dynamics(x::ElementOfLGSSM)
    return Zygote.literal_getfield(x, Val(:emission))
end

@inline function Base.getindex(model::LGSSM, n::Int)
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



"""
    marginals(model::LGSSM)

Compute the complete marginals at each point in time. These are returned as a `Vector` of
length `length(model)`, each element of which is a dense `Gaussian`.
"""
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



"""
    marginals_diag(model::LGSSM)

Compute the diagonal of the marginals at each point in time. These are returned as a
`Vector` of length `length(model)`, each element of which is a diagonal `Gaussian`.
"""
function marginals_diag(model::LGSSM)
    return scan_emit(step_marginals_diag, model, x0(model), eachindex(model))[1]
end

step_marginals_diag(x::Gaussian, model) = step_marginals_diag(ordering(model), x, model)

function step_marginals_diag(::Forward, x::Gaussian, model)
    xp = predict(x, transition_dynamics(model))
    y = predict_marginals(xp, emission_dynamics(model))
    return y, xp
end

function step_marginals_diag(::Reverse, x::Gaussian, model)
    y = predict_marginals(x, emission_dynamics(model))
    xp = predict(x, transition_dynamics(model))
    return y, xp
end



# Compute the log marginal likelihood of the observations `y`.

function Stheno.logpdf(model::LGSSM, y::AbstractVector{<:Union{AbstractVector, <:Real}})
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
    xf, lml = posterior_and_lml(x, emission_dynamics(model), y)
    xp = predict(xf, transition_dynamics(model))
    return xf, xp
end



# Construct the posterior model.

function posterior(prior::LGSSM, y::AbstractVector)
    new_trans, xf = _a_bit_of_posterior(prior, y)
    A = map(x -> x.A, new_trans)
    a = map(x -> x.a, new_trans)
    Q = map(x -> x.Q, new_trans)
    return LGSSM(GaussMarkovModel(reverse(ordering(prior)), A, a, Q, xf), prior.emissions)
end

function _a_bit_of_posterior(prior, y)
    return scan_emit(step_posterior, zip(prior, y), x0(prior), eachindex(prior))
end
step_posterior(xf::Gaussian, (prior, y)) = step_posterior(ordering(prior), xf, (prior, y))

function step_posterior(::Forward, xf::Gaussian, (prior, y))
    t = transition_dynamics(prior)
    xp = predict(xf, t)
    new_dynamics = invert_dynamics(xf, xp, t)
    xf, _ = posterior_and_lml(xp, emission_dynamics(prior), y)
    return new_dynamics, xf
end

function step_posterior(::Reverse, x::Gaussian, (prior, y))
    xf, _ = posterior_and_lml(x, emission_dynamics(prior), y)
    t = transition_dynamics(prior)
    xp = predict(xf, t)
    return invert_dynamics(xp, xf, t), xp
end

# inlining for the benefit of type inference. Needed in at least julia-1.5.3.
@inline function invert_dynamics(xf::Gaussian, xp::Gaussian, prior::SmallOutputLGC)
    A, _, _ = get_fields(prior)
    mf, Pf = get_fields(xf)
    mp, Pp = get_fields(xp)
    U = cholesky(Symmetric(Pp + ident_eps(1e-10))).U
    Gt = U \ (U' \ (A * Pf))
    return SmallOutputLGC(_collect(Gt'), mf - Gt'mp, _compute_Pf(Pf, U * Gt))
end

_compute_Pf(Pp::AbstractMatrix, B::AbstractMatrix) = Pp - B'B

ident_eps(xf::Gaussian) = ident_eps(xf, 1e-12)

ident_eps(xf, ε) = UniformScaling(convert(eltype(xf), ε))

ident_eps(ε::Real) = UniformScaling(ε)

ident_eps(x::ColVecs, ε::Real) = UniformScaling(convert(eltype(x.X), ε))

function Zygote._pullback(::NoContext, ::typeof(ident_eps), args...)
    return ident_eps(args...), nograd_pullback
end

_collect(U::Adjoint{<:Any, <:Matrix}) = collect(U)
_collect(U::SMatrix) = U



# AD stuff. No need to understand this unless you're really plumbing the depths...

function get_adjoint_storage(
    x::LGSSM, n::Int, Δx::NamedTuple{(:ordering, :transition, :emission)},
)
    return (
        transitions = get_adjoint_storage(x.transitions, n, Δx.transition),
        emissions = get_adjoint_storage(x.emissions, n, Δx.emission)
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
