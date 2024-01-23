struct PosteriorLTISDE{Tprior<:LTISDE, Tdata} <: AbstractGP
    prior::Tprior
    data::Tdata
end

# Avoids method ambiguity.
posterior(fx::FiniteLTISDE, y::AbstractVector) = _posterior(fx, y)
posterior(fx::FiniteLTISDE, y::AbstractVector{<:Real}) = _posterior(fx, y)

_posterior(fx, y) = PosteriorLTISDE(fx.f, (y=y, x=fx.x, Σy=fx.Σy))

const FinitePosteriorLTISDE = FiniteGP{<:PosteriorLTISDE}

function AbstractGPs.cov(fx::FinitePosteriorLTISDE)
    @error "Intentionally not implemented. Please don't try to explicitly compute this cov. matrix."
end

function AbstractGPs.marginals(fx::FinitePosteriorLTISDE)
    if fx.x != fx.f.data.x
        x, Σys, ys, _, pr_indices = build_inference_data(fx.f, fx.x)

        model = build_lgssm(fx.f.prior, x, Σys)
        Σys_new = noise_var_to_time_form(fx.x, fx.Σy)
        σ²s_pr_full = build_prediction_obs_vars(pr_indices, x, Σys_new)
        model_post = replace_observation_noise_cov(posterior(model, ys), σ²s_pr_full)
        return destructure(x, map(marginals, marginals(model_post))[pr_indices])
    else
        f = fx.f
        prior = f.prior
        x = fx.x
        data = f.data
        Σy = data.Σy
        Σy_diag = Σy.diag
        y = data.y

        Σy_new = fx.Σy

        model = build_lgssm(AbstractGPs.FiniteGP(prior, x, Σy))
        Σys_new = noise_var_to_time_form(x, Σy_new)
        ys = observations_to_time_form(x, y)
        model_post = replace_observation_noise_cov(posterior(model, ys), Σys_new)
        return destructure(x, map(marginals, marginals(model_post)))
    end
end

function AbstractGPs.mean_and_var(fx::FinitePosteriorLTISDE)
    ms = marginals(fx)
    return map(mean, ms), map(var, ms)
end

AbstractGPs.mean(fx::FinitePosteriorLTISDE) = mean_and_var(fx)[1]

AbstractGPs.var(fx::FinitePosteriorLTISDE) = mean_and_var(fx)[2]

function AbstractGPs.rand(rng::AbstractRNG, fx::FinitePosteriorLTISDE)
    x, Σys, ys, _, pr_indices = build_inference_data(fx.f, fx.x)

    # Get time-form observation vars and observations.
    Σys_pr = noise_var_to_time_form(fx.x, fx.Σy)

    model = build_lgssm(fx.f.prior, x, Σys)
    σ²s_pr_full = build_prediction_obs_vars(pr_indices, x, Σys_pr)
    model_post = replace_observation_noise_cov(posterior(model, ys), σ²s_pr_full)
    return destructure(x, rand(rng, model_post)[pr_indices])
end

AbstractGPs.rand(fx::FinitePosteriorLTISDE) = rand(Random.GLOBAL_RNG, fx)

function AbstractGPs.logpdf(fx::FinitePosteriorLTISDE, y_pr::AbstractVector{<:Real})

    x, Σys, ys, tr_indices, pr_indices = build_inference_data(
        fx.f, fx.x, fx.Σy, fill(missing, length(y_pr)),
    )

    # Get time-form observation vars and observations.
    Σys_pr = noise_var_to_time_form(fx.x, fx.Σy)
    ys_pr = observations_to_time_form(fx.x, y_pr)

    Σys_pr_full = build_prediction_obs_vars(pr_indices, x, Σys_pr)
    ys_pr_full = build_prediction_obs(tr_indices, pr_indices, x, ys_pr)

    model = build_lgssm(fx.f.prior, x, Σys)
    model_post = replace_observation_noise_cov(posterior(model, ys), Σys_pr_full)
    return logpdf(model_post, ys_pr_full)
end

# Join the dataset used to construct `f` and the one specified by `x_pred`, `σ²s_pred`, and
# `y_pred`. This is used in all of the inference procedures. Also provide a collection of
# indices that can be used to obtain the requested prediction locations from the joint data.

# This is the most naive way of going about this.
# The present implementation assumes that there are no overlapping data, which will lead to
# numerical issues if violated.
function build_inference_data(
    f::PosteriorLTISDE,
    x_pred::AbstractVector,
    Σy_pred::AbstractMatrix{<:Real},
    y_pred::AbstractVector{<:Union{Missing, Real}},
)
    d = f.data
    return merge_datasets((d.x, x_pred), (d.Σy, Σy_pred), (d.y, y_pred))
end

function merge_datasets(
    (x1, x2)::Tuple{AbstractVector, AbstractVector},
    (Σy1, Σy2)::Tuple{AbstractMatrix{<:Real}, AbstractMatrix{<:Real}},
    (y1, y2)::Tuple{
        AbstractVector{<:Union{Missing, Real}}, AbstractVector{<:Union{Missing, Real}},
    },
)
    # Merge and sort the inputs temporally.
    x_raw = merge_inputs(x1, x2)
    sort_indices, x = sort_in_time(x_raw)

    # Merge and sort the observation covariances temporally.
    Σys_raw = vcat(noise_var_to_time_form(x1, Σy1), noise_var_to_time_form(x2, Σy2))
    Σys = Σys_raw[sort_indices]

    # Merge and sort the observations temporally.
    ys_1 = observations_to_time_form(x1, y1)
    ys_2 = observations_to_time_form(x2, y2)
    ys_raw = vcat(ys_1, ys_2)
    ys = ys_raw[sort_indices]

    # The last length(x_pred) indices belong to the predictions.
    x1_indices = sortperm(sort_indices)[1:length(ys_1)]
    x2_indices = sortperm(sort_indices)[end-length(ys_2) + 1:end]

    return x, Σys, ys, x1_indices, x2_indices
end

# If no observations or variances are provided, make the observations arbitrary and the
# variances very large to simulate missing data.
function build_inference_data(f::PosteriorLTISDE, x_pred::AbstractVector)
    Σy_pred = Diagonal(fill(_large_var_const(), length(x_pred)))
    y_pred = fill(missing, length(x_pred))
    return build_inference_data(f, x_pred, Σy_pred, y_pred)
end


# Functions that make predictions at new locations require missings to be placed at the
# locations of the training data.
function build_prediction_obs_vars(
    pr_indices::AbstractVector{<:Integer},
    x_full::AbstractVector,
    Σys_pr::AbstractVector,
)
    σ²s_pr_full = get_zeros(x_full)
    σ²s_pr_full[pr_indices] .= Σys_pr
    return σ²s_pr_full
end

get_zeros(x::AbstractVector{T}) where {T<:Real} = zeros(T, length(x))

function build_prediction_obs(
    tr_indices::AbstractVector{<:Integer},
    pr_indices::AbstractVector{<:Integer},
    x_full::AbstractVector,
    y_pr::AbstractVector{T},
) where {T}
    y_pr_full = Vector{Union{Missing, T}}(undef, length(get_times(x_full)))
    y_pr_full[tr_indices] .= missing
    y_pr_full[pr_indices] .= y_pr
    return y_pr_full
end
