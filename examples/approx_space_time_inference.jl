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

# Construct a rectilinear grid of points in space and time.
# Exact inference only works for such grids.
# Times must be increasing, points in space can be anywhere.
N = 50;
T = 1_000;
points_in_space = collect(range(-3.0, 3.0; length=N));
points_in_time = RegularSpacing(0.0, 0.01, T);
x = RectilinearGrid(points_in_space, points_in_time);

# Generate some synthetic data from the GP under a small amount of observation noise.
y = rand(f(x, 1e-4));

# Spatial pseudo-point locations.
z_r = range(-3.0, 3.0; length=15);

# Locations in space at which to make predictions. Assumed to be the same at each point in
# time, but this assumption could easily be relaxed.
N_pr = 150;
x_r_pr = range(-5.0, 5.0; length=N_pr);

# Compute the approximate posterior marginals.
f_post_marginals = approx_posterior_marginals(dtc, f(x, 1e-4), y, z_r, x_r_pr);
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
