abstract type AbstractLGSSM end



# Compute the time-marginals of the output of the model.

function Stheno.marginals(model::AbstractLGSSM)
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

function Stheno.logpdf(model::AbstractLGSSM, y::AbstractVector)
    return sum(scan_emit(step_logpdf, zip(model, y), x0(model), eachindex(model))[1])
end

step_logpdf(x::Gaussian, model) = step_logpdf(ordering(model), x, model)

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

function _filter(model::AbstractLGSSM, y::AbstractVector)
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



# Draw a sample from the model.

function Stheno.rand(rng::AbstractRNG, model::AbstractLGSSM)
    return nothing
end



# Construct the posterior model.

function posterior(model::LGSSM, y::AbstactVector)

end
