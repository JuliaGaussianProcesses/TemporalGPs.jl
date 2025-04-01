# This is an extended version of approx_space_time_inference.jl. It combines it with
# Optim + ParameterHandling + Mooncake to learn the kernel parameters.
# If you understand how to use Optim + ParameterHandling + Mooncake for an AbstractGP,
# e.g. that shown on the README for this package, and how approx_space_time_inference.jl
# works, then you should understand this file.

using AbstractGPs
using KernelFunctions
using TemporalGPs

using TemporalGPs: Separable, approx_posterior_marginals, RegularInTime

# Load standard packages from the Julia ecosystem
using ADTypes # Way to specify algorithmic differentiation backend.
using Optim # Standard optimisation algorithms.
using ParameterHandling # Helper functionality for dealing with model parameters.
import Mooncake # Algorithmic differentiation.

using ParameterHandling: flatten

# Declare model parameters using `ParameterHandling.jl` types.
flat_initial_params, unflatten = flatten((
    var_kernel = positive(0.6),
    λ_space = positive(0.5),
    λ_time = positive(0.1),
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


# Construct inputs. Spatial locations change at each point in time.
# Also works with RectilinearGrids of inputs.
# Times must be increasing, points in space can be anywhere.
N = 50;
T = 1000;
points_in_space = [randn(N) for _ in 1:T];
points_in_time = RegularSpacing(0.0, 0.1, T);
x = RegularInTime(points_in_time, points_in_space);

# Since it's not straightforward to generate samples from this GP at `x`, use a known
# function, under a bit of iid noise.
xs = collect(x);
y = sin.(first.(xs)) .+ cos.(last.(xs)) + sqrt.(params.var_noise) .* randn(length(xs));

# Spatial pseudo-point inputs.
z_r = collect(range(-3.0, 3.0; length=5));

function objective(flat_params)
    params = unpack(flat_params)
    f = build_gp(params)
    return -elbo(f(x, params.var_noise), y, z_r)
end

@info "running objective"
@show objective(flat_initial_params)

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
final_params = unpack(training_results.minimizer);
# final_params = unpack(flat_initial_params);

# Locations in space at which to make predictions. Assumed to be the same at each point in
# time, but this assumption could easily be relaxed.
N_pr = 150;
x_r_pr = range(-5.0, 5.0; length=N_pr);

# Compute the approximate posterior marginals.
fx_final = build_gp(final_params)(x, final_params.var_noise)
f_post_marginals = approx_posterior_marginals(dtc, fx_final, y, z_r, x_r_pr);
m_post_marginals = mean.(f_post_marginals);
σ_post_marginals = std.(f_post_marginals);

# Visualise the posterior marginals. We don't do this during in CI because it causes
# problems.
if get(ENV, "TESTING", "FALSE") == "FALSE"
    using Plots
    savefig(
        plot(
            heatmap(reshape(m_post_marginals, N_pr, T)),
            heatmap(reshape(σ_post_marginals, N_pr, T));
            layout=(1, 2),
        ),
        "approx_space_time_learning.png",
    );
end
