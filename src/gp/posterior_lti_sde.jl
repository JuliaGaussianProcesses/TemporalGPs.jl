struct PosteriorLTISDE{Tprior<:LTISDE, Tdata} <: AbstractGP
    prior::Tprior
    data::Tdata
end

# Avoids method ambiguity.
posterior(fx::FiniteLTISDE, y::AbstractVector) = _posterior(fx, y)
posterior(fx::FiniteLTISDE, y::AbstractVector{<:Real}) = _posterior(fx, y)

_posterior(fx, y) = PosteriorLTISDE(fx.f, (y=y, x=fx.x, Σy=fx.Σy))

const FinitePosteriorLTISDE = FiniteGP{<:PosteriorLTISDE}

AbstractGPs.mean(fx::FinitePosteriorLTISDE) = mean.(marginals(fx))

function AbstractGPs.cov(fx::FinitePosteriorLTISDE)
    @error "Intentionally not implemented. Please don't try to explicitly compute this cov. matrix."
end

function AbstractGPs.marginals(fx::FinitePosteriorLTISDE)
    x, y, σ²s, pr_indices = build_inference_data(fx.f, fx.x)

    model = build_lgssm(fx.f.prior(x, σ²s))
    σ²s_pr_full = build_prediction_obs_vars(pr_indices, x, fx.Σy.diag)
    model_post = replace_observation_noise_cov(posterior(model, y), σ²s_pr_full)
    return map(marginals, marginals(model_post)[pr_indices])
end

function AbstractGPs.rand(rng::AbstractRNG, fx::FinitePosteriorLTISDE)
    x, y, σ²s, pr_indices = build_inference_data(fx.f, fx.x)

    model = build_lgssm(fx.f.prior(x, σ²s))
    σ²s_pr_full = build_prediction_obs_vars(pr_indices, x, fx.Σy.diag)
    model_post = replace_observation_noise_cov(posterior(model, y), σ²s_pr_full)
    return rand(rng, model_post)[pr_indices]
end

AbstractGPs.rand(fx::FinitePosteriorLTISDE) = rand(Random.GLOBAL_RNG, fx)

function AbstractGPs.logpdf(fx::FinitePosteriorLTISDE, y_pr::AbstractVector{<:Real})
    x, y, σ²s, pr_indices = build_inference_data(fx.f, fx.x, fx.Σy.diag, y_pr)

    σ²s_pr_full = build_prediction_obs_vars(pr_indices, x, fx.Σy.diag)
    y_pr_full = build_prediction_obs(pr_indices, x, y_pr)

    model = build_lgssm(fx.f.prior(x, σ²s))
    model_post = replace_observation_noise_cov(posterior(model, y), σ²s_pr_full)
    return logpdf(model_post, y_pr_full)
end

# Join the dataset used to construct `f` and the one specified by `x_pred`, `σ²s_pred`, and
# `y_pred`. This is used in all of the inference procedures. Also provide a collection of
# indices that can be used to obtain the requested prediction locations from the joint data.

# This is the most naive way of going about this.
# The present implementation assumes that there are no overlapping data, which will lead to
# numerical issues if violated.
function build_inference_data(
    f::PosteriorLTISDE,
    x_pred::AbstractVector{<:Real},
    σ²s_pred::AbstractVector,
    y_pred::AbstractVector,
)

    # Pull out the input data.
    x_cond = f.data.x
    x_raw = vcat(x_cond, x_pred)

    # Pull out the observations and create arbitrary fake observations at prediction locs.
    y_cond = f.data.y
    y_raw = vcat(y_cond, fill(missing, length(x_pred)))

    # Pull out obs. noise variance and make it really large for prediction locations.
    σ²s_cond = diag(f.data.Σy)
    σ²s_raw = vcat(σ²s_cond, σ²s_pred)

    # Put all of the data in order.
    idx = sortperm(x_raw)
    x = x_raw[idx]
    y = y_raw[idx]
    σ²s = σ²s_raw[idx]

    # The last length(x_pred) indices belong to the predictions.
    pr_indices = sortperm(idx)[end-length(x_pred) + 1:end]

    return x, y, σ²s, pr_indices
end

# If no observations or variances are provided, make the observations arbitrary and the
# variances very large to simulate missing data.
function build_inference_data(f::PosteriorLTISDE, x_pred::AbstractVector{<:Real})
    σ²s_pred = fill(convert(eltype(f.data.Σy), _large_var_const()), length(x_pred))
    y_pred = fill(missing, length(x_pred))
    return build_inference_data(f, x_pred, σ²s_pred, y_pred)
end

# Functions that make predictions at new locations require missings to be placed at the
# locations of the training data.
function build_prediction_obs_vars(
    pr_indices::AbstractVector{<:Integer},
    x_full::AbstractVector{<:Real},
    σ²s_pr::AbstractVector,
)
    σ²s_pr_full = zeros(length(x_full))
    σ²s_pr_full[pr_indices] .= σ²s_pr
    return σ²s_pr_full
end

function build_prediction_obs(
    pr_indices::AbstractVector{<:Integer},
    x_full::AbstractVector{<:Real},
    y_pr::AbstractVector,
)
    y_pr_full = Vector{Union{Missing, eltype(y_pr)}}(undef, length(x_full))
    y_pr_full[pr_indices] .= y_pr
    y_pr_full[setdiff(eachindex(x_full), pr_indices)] .= missing
    return y_pr_full
end
