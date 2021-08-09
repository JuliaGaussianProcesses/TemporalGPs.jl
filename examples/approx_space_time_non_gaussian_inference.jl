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

data = CSV.read(
    joinpath(datadep"tree-ring", "TRI2TU-data.csv"), DataFrame;
    header=[:x, :y],
)

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
