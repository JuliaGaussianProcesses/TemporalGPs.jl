# This is an extended version of exact_time_inference.jl. It combines it with
# Optim + ParameterHandling + Mooncake to learn the kernel parameters.
# Each of these other packages know nothing about TemporalGPs, they're just general-purpose
# packages which play nicely with TemporalGPs (and AbstractGPs).

using AbstractGPs
using TemporalGPs

# Load standard packages from the Julia ecosystem
using ADTypes # Way to specify algorithmic differentiation backend.
using Optim # Standard optimisation algorithms.
using ParameterHandling # Helper functionality for dealing with model parameters.
import Mooncake # Algorithmic differentiation.

# Declare model parameters using `ParameterHandling.jl` types.
# var_kernel is the variance of the kernel, λ the inverse length scale, and var_noise the
# variance of the observation noise. Note that they're all constrained to be positive.
flat_initial_params, unpack = ParameterHandling.value_flatten((
    mean = 3.0,
    var_kernel = positive(0.6),
    λ = positive(0.1),
    var_noise = positive(2.0),
));

# Pull out the raw values.
params = unpack(flat_initial_params);

# Functionality to load build a TemporalGPs.jl GP given the model parameters.
# Specifying SArrayStorage ensures that StaticArrays.jl is used to represent model
# parameters under the hood, which enables very strong peformance.
function build_gp(params)
    k = params.var_kernel * Matern52Kernel() ∘ ScaleTransform(params.λ)
    return to_sde(GP(params.mean, k), SArrayStorage(Float64))
end

# Specify a collection of inputs. Must be increasing.
T = 1_000_000;
x = TemporalGPs.RegularSpacing(0.0, 1e-4, T);

# Generate some noisy synthetic data from the GP.
f = build_gp(params)
y = rand(f(x, params.var_noise));

# Specify an objective function for Optim to minimise in terms of x and y.
# We choose the usual negative log marginal likelihood (NLML).
function objective(flat_params)
    params = unpack(flat_params)
    f = build_gp(params)
    return -logpdf(f(x, params.var_noise), y)
end

# Optimise using Optim. Mooncake takes a little while to compile.
training_results = Optim.optimize(
    objective,
    flat_initial_params .+ randn.(), # Perturb the parameters to make learning non-trivial
    BFGS(
        alphaguess = Optim.LineSearches.InitialStatic(scaled=true),
        linesearch = Optim.LineSearches.BackTracking(),
    ),
    Optim.Options(show_trace = true);
    inplace=false,
    autodiff=AutoMooncake(; config=nothing),
);

# Extracting the final values of the parameters. Should be moderately close to truth.
final_params = unpack(training_results.minimizer)

# Construct the posterior as per usual.
f_final = build_gp(final_params)
f_post = posterior(f_final(x, final_params.var_noise), y);

# Specify some locations at which to make predictions.
T_pr = 1_200_000;
x_pr = TemporalGPs.RegularSpacing(0.0, 1e-4, T_pr);

# Compute the exact posterior marginals at `x_pr`.
f_post_marginals = marginals(f_post(x_pr));
m_post_marginals = mean.(f_post_marginals);
σ_post_marginals = std.(f_post_marginals);

# Generate a few posterior samples. Not fantastically-well optimised at present.
f_post_samples = [rand(f_post(x_pr)) for _ in 1:5];

# Visualise the posterior. The if block is just to prevent it running in CI.
if get(ENV, "TESTING", "FALSE") == "FALSE"
    using Plots
    plt = plot();
    scatter!(plt, x, y; label="", markersize=0.1, alpha=0.1);
    plot!(plt, f_post(x_pr); ribbon_scale=3.0, label="");
    plot!(plt, x_pr, f_post_samples; color=:red, label="");
    savefig(plt, "exact_time_learning.png");
end
