# Get hold of the dataset used in this example.

using CSV
using DataDeps
using DataFrames

register(DataDep(
    "tree-ring",
    "Downloading tree ring data",
    "https://willtebbutt.github.io/resources/TRI2TU-data.csv",
));

raw_data = CSV.read(
    joinpath(datadep"tree-ring", "TRI2TU-data.csv"), DataFrame; header=[:x, :y],
);

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

data = construct_count_matrix(10, Matrix(raw_data));

# Doing approximate inference in space-time GPs using pseudo-points is, as always, similar
# to performing exact inference. The primary difference is that `rand` isn't available,
# and the `approx_posterior_marginals` function must be used to query the approximate
# posterior. This is hopefully a temporary solution, so should change at some point in the
# future.

using AbstractGPs
using ConjugateComputationVI
using Distributions
using KernelFunctions
using StatsFuns
using TemporalGPs
using Zygote

using ConjugateComputationVI:
    approx_posterior,
    build_reconstruction_term,
    GaussHermiteQuadrature,
    optimise_approx_posterior,
    UnivariateFactorisedLikelihood

# Load up the separable kernel from TemporalGPs. You need to use this to tell TemporalGPs
# that you're using a separable kernel (it's not enough just to use a kernel which
# happens to be separable).
# RegularInTime is a data structure for inputs which allows for different spatial locations
# at each point in time, and can be used with the approximate inference scheme presented
# here.
using TemporalGPs: Separable, RectilinearGrid, approx_posterior_marginals

# Adjoint for the Poisson logpdf.
# log(λ^x exp(-λ) / x!) =
# x log(λ) - λ - log(x!)
# dλ = x / λ - 1
Zygote.@adjoint function StatsFuns.poislogpdf(λ::Float64, x::Union{Float64, Int})
    function poislogpdf_pullback(Δ::Real)
        return Δ * (x / λ - 1), nothing
    end
    return StatsFuns.poislogpdf(λ, x), poislogpdf_pullback
end

# Convert data into format suitable for TemporalGPs.
T = size(data, 1);
N = size(data, 2);
points_in_space = map(float, collect(1:N)) .* 10 / N;
points_in_time = RegularSpacing(0.0, 10.0 / T, T);
x = RectilinearGrid(points_in_space, points_in_time);
y = map(float, vec(data));

# Specify a LatentGP in which to perform inference.
gp = to_sde(GP(Separable(SEKernel(), Matern52Kernel())), ArrayStorage(Float64));
lik = UnivariateFactorisedLikelihood(f -> Poisson(exp(f)));
latent_gp = LatentGP(gp, lik, 1e-9);

# Utilise some internals from CVI to perform inference. Possibly these should be more
# nicely wrapped in a user-facing API.
r = build_reconstruction_term(GaussHermiteQuadrature(10), latent_gp, y);
η1, η2, iters, delta = optimise_approx_posterior(
    gp, x, zeros(length(x)), -ones(length(x)), r, 1; tol=1e-4,
)
approx_post = approx_posterior(gp, x, η1, η2)

using Plots

N_high_res = 50;
T_high_res = 1_000;
points_in_space_high_res = collect(
    range(minimum(points_in_space), maximum(points_in_space); length=N_high_res),
);
points_in_time_high_res = collect(
    range(minimum(points_in_time), maximum(points_in_time); length=T_high_res),
);
x_high_res = RectilinearGrid(points_in_space, points_in_time_high_res);
approx_post_marginals = reshape(marginals(approx_post(x_high_res)), N, T_high_res);

x_scaling = length(points_in_time_high_res) / 1000;
y_scaling = length(points_in_space) / 500;

hmp = heatmap(mean.(approx_post_marginals));
sct = scatter(raw_data.x .* x_scaling, raw_data.y .* y_scaling; color=:white, markersize=1);
plot(hmp, sct; layout=(2, 1))
