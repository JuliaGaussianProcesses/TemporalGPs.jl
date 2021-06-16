# This is an extended version of exact_time_inference.jl. It just combines it with
# Optim + ParameterHandling + Zygote to learn the kernel parameters.
# If you understand how to use Optim + ParameterHandling + Zygote for an AbstractGP,
# e.g. that shown on the README for this package, and how exact_time_inference.jl
# works, then you should understand this file.

using AbstractGPs
using TemporalGPs

# Load up the separable kernel from TemporalGPs.
using TemporalGPs: RegularSpacing

# Load standard packages from the Julia ecosystem
using Optim # Standard optimisation algorithms.
using ParameterHandling # Helper functionality for dealing with model parameters.
using Zygote # Algorithmic Differentiation

using ParameterHandling: flatten

# Declare model parameters using `ParameterHandling.jl` types.
flat_initial_params, unflatten = flatten((
    var_kernel = positive(0.6),
    λ = positive(0.1),
    var_noise = positive(0.1),
));

# Construct a function to unpack flattened parameters and pull out the raw values.
unpack = ParameterHandling.value ∘ unflatten;
params = unpack(flat_initial_params);

function build_gp(params)
    k = params.var_kernel * Matern52Kernel() ∘ ScaleTransform(params.λ)
    return to_sde(GP(k), SArrayStorage(Float64))
end

# Specify a collection of inputs. Must be increasing.
T = 1_000_000;
x = RegularSpacing(0.0, 1e-4, T);

# Generate some synthetic data from the GP.
y = rand(f(x, params.var_noise));

# Specify an objective function for Optim to minimise in terms of x and y.
# We choose the usual negative log marginal likelihood (NLML).
function objective(params)
    f = build_gp(params)
    return -logpdf(f(x, params.var_noise), y)
end

# Optimise using Optim. Takes a little while to compile because Zygote.
training_results = Optim.optimize(
    objective ∘ unpack,
    θ -> only(Zygote.gradient(objective ∘ unpack, θ)),
    flat_initial_params + randn(3), # Add some noise to make learning non-trivial
    BFGS(
        alphaguess = Optim.LineSearches.InitialStatic(scaled=true),
        linesearch = Optim.LineSearches.BackTracking(),
    ),
    Optim.Options(show_trace = true);
    inplace=false,
);

# Extracting the final values of the parameters.
# Should be close to truth.
final_params = unpack(training_results.minimizer);

# Construct the posterior as per usual.
f_post = posterior(build_gp(final_params)(x, final_params.var_noise), y);

# Specify some locations at which to make predictions.
T_pr = 1_200_000;
x_pr = RegularSpacing(0.0, 1e-4, T_pr);

# Compute the exact posterior marginals at `x_pr`.
f_post_marginals = marginals(f_post(x_pr));
m_post_marginals = mean.(f_post_marginals);
σ_post_marginals = std.(f_post_marginals);

# Generate a few posterior samples. Not fantastically-well optimised at present.
f_post_samples = [rand(f_post(x_pr)) for _ in 1:5];

# Visualise the posterior marginals. We don't do this during in CI.
if get(ENV, "TESTING", "FALSE") == "FALSE"
    using Plots
    plt = plot();
    scatter!(plt, x, y; label="", markersize=0.1, alpha=0.1);
    plot!(plt, f_post(x_pr); ribbon_scale=3.0, label="");
    plot!(x_pr, f_post_samples; color=:red, label="");
    savefig(plt, "posterior.png");
end
