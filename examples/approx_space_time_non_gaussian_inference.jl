# Get hold of the dataset used in this example.

using CSV
using DataDeps
using DataFrames
using Plots

register(DataDep(
    "tree-ring",
    "Downloading tree ring data",
    "https://willtebbutt.github.io/resources/TRI2TU-data.csv",
))

raw_data = CSV.read(
    joinpath(datadep"tree-ring", "TRI2TU-data.csv"), DataFrame;
    header=[:x, :y],
)

# The data must be binned to work with the Point process. The finer the binning, the closer
# the approximation to the process of interest.
function construct_count_matrix(bin_width::Integer, tree_locations)

    # Compute bin sizes and ensure that `bin_width` divides 1000 and 500 exactly.
    x_len, y_len = Int(div(1000, bin_width)), Int(div(500, bin_width))
    @assert x_len * bin_width == 1000
    @assert y_len * bin_width == 500
    count_matrix = fill(0, x_len, y_len)

    # Fill the count matrix.
    for n in 1:size(tree_locations, 1)
        x, y = tree_locations[n, 1], tree_locations[n, 2]
        x_bin, y_bin = Int(floor(x / bin_width)) + 1, Int(floor(y / bin_width)) + 1
        count_matrix[x_bin, y_bin] += 1
    end

    return collect(count_matrix)
end

data = construct_count_matrix(1, Matrix(raw_data));

# Doing approximate inference in space-time GPs using pseudo-points is, as always, similar
# to performing exact inference. The primary difference is that `rand` isn't available,
# and the `approx_posterior_marginals` function must be used to query the approximate
# posterior. This is hopefully a temporary solution, so should change at some point in the
# future.

using AbstractGPs
using ConjugateComputationVI
using TemporalGPs

# Load up the separable kernel from TemporalGPs. You need to use this to tell TemporalGPs
# that you're using a separable kernel (it's not enough just to use a kernel which
# happens to be separable).
# RegularInTime is a data structure for inputs which allows for different spatial locations
# at each point in time, and can be used with the approximate inference scheme presented
# here.
using TemporalGPs: Separable, RegularInTime, approx_posterior_marginals

# Specify a separable kernel.
# The first argument is always the kernel over space, the second the kernel over time.
# You can also use sums of separble kernels.
k = Separable(SEKernel(), Matern52Kernel());

# Build a GP, and convert it to an SDE as per usual.
# Use `ArrayStorage`, not `SArrayStorage`, for these kinds of GPs.
f = to_sde(GP(k), ArrayStorage(Float64));

# Convert data into format suitable for TemporalGPs.
points_in_space = data.x
points_in_time = data.y
N = 50;
T = 1_000;
points_in_space = collect(range(-3.0, 3.0; length=N));
points_in_time = RegularSpacing(0.0, 0.01, T);
x = RectilinearGrid(points_in_space, points_in_time);
