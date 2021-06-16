using AbstractGPs
using TemporalGPs

# Doing approximate inference in space-time GPs using pseudo-points is, as always, similar
# to performing exact inference. The primary difference is that `rand` isn't available,
# and the `approx_posterior_marginals` function must be used to query the approximate
# posterior. This is hopefully a temporary solution, so should change at some point in the
# future.

# Load up the separable kernel from TemporalGPs. You need to use this to tell TemporalGPs
# that you're using a separable kernel (it's not enough just to use a kernel which
# happens to be separable).
using TemporalGPs: Separable, RectilinearGrid, approx_posterior_marginals

# Specify a separable kernel.
# The first argument is always the kernel over space, the second the kernel over time.
# You can also use sums of separble kernels.
k = Separable(SEKernel(), Matern52Kernel());

# Build a GP, and convert it to an SDE as per usual.
# Use `ArrayStorage`, not `SArrayStorage`, for these kinds of GPs.
f = to_sde(GP(k), ArrayStorage(Float64));

# Construct inputs. Spatial locations change at each point in time.
# Also works with RectilinearGrids of inputs.
# Exact inference only works for such grids.
# Times must be increasing, points in space can be anywhere.
N = 50;
T = 1_000;
points_in_space = [randn(N) for _ in 1:T];
points_in_time = RegularSpacing(0.0, 0.1, T);
x = RegularInTime(points_in_time, points_in_space);

# Since it's not straightforward to generate samples from this GP at `x`, use a known
# function, under a bit of iid noise.
xs = collect(x);
y = sin.(first.(xs)) .+ cos.(last.(xs)) + sqrt.(0.1) .* randn(length(xs));

# Spatial pseudo-point locations.
z_r = range(-3.0, 3.0; length=5);

# Locations in space at which to make predictions. Assumed to be the same at each point in
# time, but this assumption could easily be relaxed.
N_pr = 150;
x_r_pr = range(-5.0, 5.0; length=N_pr);

# Compute the approximate posterior marginals.
f_post_marginals = approx_posterior_marginals(dtc, f(x, 0.1), y, z_r, x_r_pr);
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
        "posterior.png",
    );
end
