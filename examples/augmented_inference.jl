using AbstractGPs
using AugmentedGPLikelihoods
using TemporalGPs
using Distributions
using StatsFuns: logistic

# In this example we are showing how to work with non-Gaussian likelihoods,
# here a binary classification problem with a logistic link,
# by using variable augmentations as in [Galy-Fajou et al, 2020](https://arxiv.org/abs/2002.114510).
# We base our example on the "exact_time_inference" example

# Load up the separable kernel from TemporalGPs.
using TemporalGPs: RegularSpacing

# Build a GP as per usual, and wrap it inside a TemporalGPs.jl object.
f_raw = GP(Matern52Kernel());
f = to_sde(f_raw, SArrayStorage(Float64));

# Specify a collection of inputs. Must be increasing.
T = 1_000;
x = RegularSpacing(0.0, 1e-1, T);

# Generate some synthetic data from the GP and get a random binary output
σ²_noise = 0.01;
f_true = rand(f(x, σ²_noise));
lik = BernoulliLikelihood() # Likelihood with a logistic link
y = rand(lik(f_true));

function compute_optimal_expectation(f, x, g, β, γ; n_iter=5)
    T = length(x)
    qω = init_aux_posterior(lik, y)
    for i in 1:n_iter
        Λ = expected_auglik_precision(lik, qω, y)
        h = expected_auglik_potential(lik, qω, y)
        qf = marginals(posterior(f(x, inv.(Λ)), h)(x))
        aux_posterior!(qω, qf) # Update the posterior
    end
    return qω
end

qω = compute_optimal_expectation(f, x, g, β, γ)
h, Λ = expected_auglik_potential_and_precision(lik, qω, y) 
f_post = posterior(f(x, inv.(Λ)), h)

# Specify some locations at which to make predictions.
T_pr = 1_200_000;
x_pr = RegularSpacing(0.0, 1e-4, T_pr);

# Compute the exact posterior marginals at `x_pr`.
f_post_marginals = marginals(f_post(x_pr));
m_post_marginals = mean.(f_post_marginals);
σ_post_marginals = std.(f_post_marginals);

# Generate a few posterior samples. Not fantastically-well optimised at present.
f_post_samples = [rand(f_post(x_pr)) for _ in 1:5];

# Visualise the posterior. The if block is just to prevent it running in CI.
if get(ENV, "TESTING", "FALSE") == "FALSE"
    using Plots
    plt = plot()
    plot!(plt, f_post(x_pr); ribbon_scale=3.0, label="");
    plot!(plt, x_pr, f_post_samples; color=:red, alpha=0.3, label="");
    plot!(plt, x, f_true; label="", lw=2.0, color=:blue); # Plot the true latent GP on top 
    scatter!(plt, x, y; label="", markersize=1.0, alpha=1.0); # Plot the data
    savefig(plt, "posterior.png");
end
