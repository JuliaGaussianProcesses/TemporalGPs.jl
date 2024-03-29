# Working with spatio-temporal GPs in TemporalGPs.jl mostly looks like working with any
# other GP. The main differences are to do with what you're allowed to specify as a kernel,
# and what kinds of inputs work.

using AbstractGPs
using TemporalGPs

# Load up the separable kernel from TemporalGPs. You need to use this to tell TemporalGPs
# that you're using a separable kernel (it's not enough just to use a kernel which
# happens to be separable).
# Similarly, you need to use a RectilinearGrid to specify the grid of inputs -- it's not
# enough to ensure that the inputs happen to live on a grid.
using TemporalGPs: Separable, RectilinearGrid

# Specify a separable kernel.
# The first argument is always the kernel over space, the second the kernel over time.
# You can also use weighted sums of separble kernels.
k = Separable(SEKernel(), Matern52Kernel());

# Build a GP, and convert it to an SDE as per usual.
# Use `ArrayStorage`, not `SArrayStorage`, for these kinds of GPs.
f = to_sde(GP(k), ArrayStorage(Float64));

# Construct a rectilinear grid of points in space and time.
# Exact inference only works for such grids.
# Times must be increasing, points in space can be anywhere, and can be multi-dimensional.
# For multi-dimensional spatial-locations, you would probably want to use a ColVecs.
N = 50;
T = 1_000;
points_in_space = collect(range(-3.0, 3.0; length=N));
points_in_time = RegularSpacing(0.0, 0.01, T);
x = RectilinearGrid(points_in_space, points_in_time);

# Generate some noisy synthetic data from the GP.
y = rand(f(x, 1e-1));

# Construct the posterior as per usual.
f_post = posterior(f(x, 1e-1), y);

# Specify some locations at which to make predictions.
# This must be another RectilinearGrid with the same spatial locations as x.
T_pr = 1200;
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
        "exact_space_time_inference.png",
    );
end
