struct PosteriorLTISDE{Tprior<:LTISDE, Tdata} <: AbstractGP
    prior::Tprior
    data::Tdata
end

function posterior(fx::FiniteLTISDE, y::AbstractVector{<:Real})
    return PosteriorLTISDE(fx.f, (y=y, x=fx.x, Σy=fx.Σy))
end

const FinitePosteriorLTISDE = FiniteGP{<:PosteriorLTISDE}

# Join the dataset used to construct `f` and the one specified by `x_pred`, `σ²s_pred`, and
# `y_pred`. This is used in all of the inference procedures. Also provide a collection of
# indices that can be used to obtain the requested prediction locations from the joint data.
function build_inference_data(
    f::PosteriorLTISDE,
    x_pred::AbstractVector{<:Real},
    σ²s_pred::AbstractVector{<:Real},
    y_pred::AbstractVector{<:Real},
)

    # Pull out the input data.
    x_cond = f.data.x
    x, idx = merge_and_sort(x_cond, x_pred)

    # Pull out the observations and create arbitrary fake observations at prediction locs.
    y_cond = f.data.y
    y_raw = vcat(y_cond, y_pred)

    # Pull out obs. noise variance and make it really large for prediction locations.
    σ²s_cond = diag(f.data.Σy)
    σ²s_raw = vcat(σ²s_cond, σ²s_pred)

    # Put all of the data in order.
    y = y_raw[idx]
    σ²s = σ²s_raw[idx]

    # The last length(x_pred) indices belong to the predictions.
    pr_indices = sortperm(idx)[end-length(x_pred) + 1:end]

    return x, y, σ²s, pr_indices
end

# Merge a and b, returning them sorted, as well as the indices required to sort them.
function merge_and_sort(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    x = vcat(a, b)
    idx = sortperm(x)
    return x[idx], idx
end

function merge_and_sort(a::RegularSpacing, b::RegularSpacing)
    if a == b
        return a, eachindex(a)
    else
        return merge_and_sort(collect(a), collect(b))
    end
end

# If no observations or variances are provided, make the observations arbitrary and the
# variances very large to simulate missing data.
function build_inference_data(f::PosteriorLTISDE, x_pred::AbstractVector{<:Real})
    σ²s_pred = fill(convert(eltype(f.data.Σy), 1_000_000_000), length(x_pred))
    y_pred = fill(convert(eltype(f.data.y), 0), length(x_pred))
    return build_inference_data(f, x_pred, σ²s_pred, y_pred)
end

function Stheno.mean(fx::FinitePosteriorLTISDE)
    return mean.(marginals(fx))
end

function Stheno.cov(fx::FinitePosteriorLTISDE)
    @error "Not implemented. Please don't try to explicitly compute this cov. matrix."
end

function Stheno.marginals(fx::FinitePosteriorLTISDE)
    x, y, σ²s, pr_indices = build_inference_data(fx.f, fx.x)

    lgssm  = build_lgssm(fx.f.prior(x, σ²s))
    _, posterior_marginals, _ = smooth(lgssm, y)
    pr_posterior_marginals = posterior_marginals[pr_indices]

    return Normal.(
        only.(getfield.(pr_posterior_marginals, :m)),
        sqrt.(only.(getfield.(pr_posterior_marginals, :P)) .+ fx.Σy.diag),
    )
end

function Stheno.rand(rng::AbstractRNG, fx::FinitePosteriorLTISDE)
    x, y, σ²s, pr_indices = build_inference_data(fx.f, fx.x)

    lgssm = build_lgssm(fx.f.prior(x, σ²s))
    fxs = posterior_rand(rng, lgssm, y)
    return fxs[pr_indices] .+ sqrt.(fx.Σy.diag) .* randn(rng, length(pr_indices))
end

Stheno.rand(fx::FinitePosteriorLTISDE) = rand(Random.GLOBAL_RNG, fx)

function Stheno.rand(rng::AbstractRNG, ft::FinitePosteriorLTISDE, N::Int)
    return hcat([rand(rng, ft) for _ in 1:N]...)
end

Stheno.rand(ft::FinitePosteriorLTISDE, N::Int) = rand(Random.GLOBAL_RNG, ft, N)

function Stheno.logpdf(fx::FinitePosteriorLTISDE, y_pr::AbstractVector{<:Real})
    x, y, σ²s, _ = build_inference_data(fx.f, fx.x, diag(fx.Σy), y_pr)

    @warn "posterior logpdf might be wrong :S probably best not to trust the results..."

    logp_prior = logpdf(fx.f.prior(fx.f.data.x, fx.f.data.Σy), fx.f.data.y)
    logp_all = logpdf(fx.f.prior(x, σ²s), y)
    return logp_all - logp_prior
end
