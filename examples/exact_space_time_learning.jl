# This is an extended version of exact_space_time_inference.jl. It combines it with
# Optim + ParameterHandling + Mooncake to learn the kernel parameters.
# If you understand how to use Optim + ParameterHandling + Mooncake for an AbstractGP,
# e.g. that shown on the README for this package, and how exact_space_time_inference.jl
# works, then you should understand this file.

using AbstractGPs
using KernelFunctions
using TemporalGPs

using TemporalGPs: Separable, RectilinearGrid

# Load standard packages from the Julia ecosystem
using ADTypes # Way to specify algorithmic differentiation backend.
using Optim # Standard optimisation algorithms.
using ParameterHandling # Helper functionality for dealing with model parameters.
import Mooncake # Algorithmic differentiation.

# Declare model parameters using `ParameterHandling.jl` types.
flat_initial_params, unflatten = ParameterHandling.flatten((
    var_kernel = positive(0.6),
    λ_space = positive(2.5),
    λ_time = positive(2.5),
    var_noise = positive(0.1),
));

# Construct a function to unpack flattened parameters and pull out the raw values.
unpack = ParameterHandling.value ∘ unflatten;
params = unpack(flat_initial_params);

function build_gp(params)
    k_space = SEKernel() ∘ ScaleTransform(params.λ_space)
    k_time = Matern52Kernel() ∘ ScaleTransform(params.λ_time)
    k = params.var_kernel * Separable(k_space, k_time)
    return to_sde(GP(k), ArrayStorage(Float64))
end


# Construct a rectilinear grid of points in space and time.
# Exact inference only works for such grids.
# Times must be increasing, points in space can be anywhere.
N = 50;
T = 500;
points_in_space = collect(range(-3.0, 3.0; length=N));
points_in_time = RegularSpacing(0.0, 0.01, T);
x = RectilinearGrid(points_in_space, points_in_time);
y = rand(build_gp(params)(x, 1e-4));

# Specify an objective function for Optim to minimise in terms of x and y.
# We choose the usual negative log marginal likelihood (NLML).
function objective(flat_params)
    params = unpack(flat_params)
    f = build_gp(params)
    return -logpdf(f(x, params.var_noise), y)
end

# Optimise using Optim.
training_results = Optim.optimize(
    objective,
    flat_initial_params + randn(4), # Add some noise to make learning non-trivial
    BFGS(
        alphaguess = Optim.LineSearches.InitialStatic(scaled=true),
        linesearch = Optim.LineSearches.BackTracking(),
    ),
    Optim.Options(show_trace = true);
    inplace=false,
    autodiff=AutoMooncake(; config=nothing),
);

# Extracting the final values of the parameters.
# Should be close to truth.
final_params = unpack(training_results.minimizer)

# Construct the posterior as per usual.
f_post = posterior(build_gp(final_params)(x, final_params.var_noise), y);

# Specify some locations at which to make predictions.
T_pr = 600;
points_in_time_pr = RegularSpacing(0.0, 0.01, T_pr);
x_pr = RectilinearGrid(points_in_space, points_in_time_pr);

# Compute the exact posterior marginals at `x_pr`.
# This isn't optimised at present, so might take a little while.
f_post_marginals = marginals(f_post(x_pr));
m_post_marginals = mean.(f_post_marginals);
σ_post_marginals = std.(f_post_marginals);

# Visualise the posterior marginals. We don't do this during in CI because it causes
# problems.
if get(ENV, "TESTING", "FALSE") == "FALSE"
    using Plots
    savefig(
        plot(
            heatmap(reshape(m_post_marginals, N, T_pr)),
            heatmap(reshape(σ_post_marginals, N, T_pr));
            layout=(1, 2),
        ),
        "exact_space_time_learning.png",
    );
end
