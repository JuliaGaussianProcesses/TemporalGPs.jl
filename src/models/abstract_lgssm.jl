abstract type AbstractLGSSM end

function Stheno.marginals(model::AbstractLGSSM)
    return scan_emit(step_marginals, model, x0(model), eachindex(model))[1]
end

function Stheno.logpdf(model::AbstractLGSSM, y::AbstractVector)
    pick_lml(((lml, _), x)) = (lml, x)
    return sum(scan_emit(
        pick_lml ∘ step_decorrelate, zip(model, y), x0(model), eachindex(model),
    )[1])
end

function decorrelate(model::AbstractLGSSM, y::AbstractVector)
    pick_α(((_, α), x)) = (α, x)
    α, _ = scan_emit(pick_α ∘ step_decorrelate, zip(model, y), x0(model), eachindex(model))
    return α
end

function _filter(model::AbstractLGSSM, y::AbstractVector)
    pick_x((_, x)) = (x, x)
    xs, _ = scan_emit(pick_x ∘ step_decorrelate, zip(model, y), x0(model), eachindex(model))
    return xs
end

function correlate(model::AbstractLGSSM, α::AbstractVector)
    pick_y(((_, y), x)) = (y, x)
    ys, _ = scan_emit(pick_y ∘ step_correlate, zip(model, α), x0(model), eachindex(model))
    return ys
end

Stheno.rand(rng::AbstractRNG, model::AbstractLGSSM) = correlate(model, rand_αs(rng, model))

function rand_αs(rng::AbstractRNG, model::AbstractLGSSM)
    return map(_ -> randn(rng, eltype(model), dim_obs(model)), 1:length(model))
end

ChainRulesCore.@non_differentiable rand_αs(::AbstractRNG, ::AbstractLGSSM)



#
# step - these are the functions to which scan_emit can be applied.
#

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

step_decorrelate(x::Gaussian, (model, y)) = step_decorrelate(ordering(model), x, (model, y))

function step_decorrelate(::Forward, x::Gaussian, (model, y))
    xp = predict(x, transition_dynamics(model))
    xf, lml, α = decorrelate(xp, emission_dynamics(model), y)
    return (lml, α), xf
end

function step_decorrelate(::Reverse, x::Gaussian, (model, y))
    xf, lml, α = decorrelate(x, emission_dynamics(model), y)
    xp = predict(xf, transition_dynamics(model))
    return (lml, α), xp
end

step_correlate(x::Gaussian, (model, α)) = step_correlate(ordering(model), x, (model, α))

function step_correlate(::Forward, x::Gaussian, (model, α))
    xp = predict(x, transition_dynamics(model))
    xf, lml, y = correlate(xp, emission_dynamics(model), α)
    return (lml, y), xf
end

function step_correlate(::Reverse, x::Gaussian, (model, α))
    xf, lml, y = correlate(x, emission_dynamics(model), α)
    xp = predict(xf, transition_dynamics(model))
    return (lml, y), xp
end

Zygote.@nograd ordering
