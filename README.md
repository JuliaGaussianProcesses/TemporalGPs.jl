# TemporalGPs

[![Build Status](https://github.com/willtebbutt/TemporalGPs.jl/workflows/CI/badge.svg)](https://github.com/willtebbutt/TemporalGPs.jl/actions)
[![Coverage Status](https://coveralls.io/repos/github/JuliaGaussianProcesses/TemporalGPs.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaGaussianProcesses/TemporalGPs.jl?branch=master)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

TemporalGPs.jl is a tool to make Gaussian processes (GPs) defined using [AbstractGPs.jl](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl/) fast for time-series. It provides a single-function public API that lets you specify that this package should perform inference, rather than AbstractGPs.jl.

[JuliaCon 2020 Talk](https://www.youtube.com/watch?v=dysmEpX1QoE)

# Installation

TemporalGPs.jl is registered, so simply type the following at the REPL:
```julia
] add AbstractGPs KernelFunctions TemporalGPs
```
While you can install TemporalGPs without AbstractGPs and KernelFunctions, in practice the latter are needed for all common tasks in TemporalGPs.

# Example Usage

Most examples can be found in the [examples](https://github.com/JuliaGaussianProcesses/TemporalGPs.jl/tree/master/examples) directory. In particular see the associated [README](https://github.com/JuliaGaussianProcesses/TemporalGPs.jl/tree/master/examples/README.md).

This is a small problem by TemporalGPs' standard. See timing results below for expected performance on larger problems.

```julia
using AbstractGPs, KernelFunctions, TemporalGPs

# Specify a AbstractGPs.jl GP as usual
f_naive = GP(Matern32Kernel())

# Wrap it in an object that TemporalGPs knows how to handle.
f = to_sde(f_naive, SArrayStorage(Float64))

# Project onto finite-dimensional distribution as usual.
# x = range(-5.0; step=0.1, length=10_000)
x = RegularSpacing(0.0, 0.1, 10_000) # Hack for Zygote.
fx = f(x, 0.1)

# Sample from the prior as usual.
y = rand(fx)

# Compute the log marginal likelihood of the data as usual.
logpdf(fx, y)

# Construct the posterior distribution over `f` having observed `y` at `x`.
f_post = posterior(fx, y)

# Compute the posterior marginals.
marginals(f_post(x))

# Draw a sample from the posterior. Note: same API as prior.
rand(f_post(x))

# Compute posterior log predictive probability of `y`. Note: same API as prior.
logpdf(f_post(x), y)
```

## Learning kernel parameters with [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl), [ParameterHandling.jl](https://github.com/invenia/ParameterHandling.jl), and [Zygote.jl](https://github.com/FluxML/Zygote.jl/)

TemporalGPs.jl doesn't provide scikit-learn-like functionality to train your model (find good kernel parameter settings).
Instead, we offer the functionality needed to easily implement your own training functionality using standard tools from the Julia ecosystem, as shown below.
```julia
# Load our GP-related packages.
using AbstractGPs
using KernelFunctions
using TemporalGPs

# Load standard packages from the Julia ecosystem
using Optim # Standard optimisation algorithms.
using ParameterHandling # Helper functionality for dealing with model parameters.
using Zygote # Algorithmic Differentiation

using ParameterHandling: flatten

# Declare model parameters using `ParameterHandling.jl` types.
flat_initial_params, unflatten = flatten((
    var_kernel = positive(0.6),
    λ = positive(2.5),
    var_noise = positive(0.1),
))

# Construct a function to unpack flattened parameters and pull out the raw values.
unpack = ParameterHandling.value ∘ unflatten
params = unpack(flat_initial_params)

function build_gp(params)
    f_naive = GP(params.var_kernel * Matern52Kernel() ∘ ScaleTransform(params.λ))
    return to_sde(f_naive, SArrayStorage(Float64))
end

# Generate some synthetic data from the prior.
const x = RegularSpacing(0.0, 0.1, 10_000)
const y = rand(build_gp(params)(x, params.var_noise))

# Specify an objective function for Optim to minimise in terms of x and y.
# We choose the usual negative log marginal likelihood (NLML).
function objective(params)
    f = build_gp(params)
    return -logpdf(f(x, params.var_noise), y)
end

# Check that the objective function works:
objective(params)

# Optimise using Optim. This optimiser often works fairly well in practice,
# but it's not going to be the best choice in all situations. Consult
# Optim.jl for more info on available optimisers and their properties.
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
)

# Extracting the final values of the parameters.
# Should be close to truth.
final_params = unpack(training_results.minimizer)
```
Once you've learned the parameters, you can use `posterior`, `marginals`, and `rand` to make posterior-predictions with the optimal parameters.

In the above example we optimised the parameters, but we could just as easily have utilised e.g. [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl) in conjunction with a prior over the parameters to perform approximate Bayesian inference in them -- indeed, [this is often a very good idea](http://proceedings.mlr.press/v118/lalchand20a/lalchand20a.pdf). We leave this as an exercise for the interested user (see e.g. the examples in [Stheno.jl](https://github.com/willtebbutt/Stheno.jl/) for inspiration).

Moreover, it should be possible to plug this into probabilistic programming framework such as `Turing` and `Soss` with minimal effort, since `f(x, params.var_noise)` is a plain old `Distributions.MultivariateDistribution`.


# Performance Optimisations

There are a couple of ways that `TemporalGPs.jl` can represent things internally. In particular, it can use regular Julia `Vector` and `Matrix` objects, or the `StaticArrays.jl` package to optimise in certain cases. The default is the former. To employ the latter, just add an extra argument to the `to_sde` function:
```julia
f = to_sde(f_naive, SArrayStorage(Float64))
```
This tells TemporalGPs that you want all parameters of `f` and anything derived from it to be a subtype of a `SArray` with element-type `Float64`, rather than (for example) a `Matrix{Float64}`s and `Vector{Float64}`. The decision made here can have quite a dramatic effect on performance, as shown in the graph below. For "larger" kernels (large sums, spatio-temporal problems), you might want to consider `ArrayStorage(Float64)` instead.



# Benchmarking Results

![](/examples/benchmarks.png)

"naive" timings are with the usual [AbstractGPs.jl](https://https://github.com/JuliaGaussianProcesses/AbstractGPs.jl/) inference routines, and is the default implementation for GPs. "lgssm" timings are conducted using `to_sde` with no additional arguments. "static-lgssm" uses the `SArrayStorage(Float64)` option discussed above.

Gradient computations use Zygote. Custom adjoints have been implemented to achieve this level of performance.



# On-going Work

- Optimisation
    + in-place implementation with `ArrayStorage` to reduce allocations
    + input data types for posterior inference - the `RegularSpacing` type is great for expressing that the inputs are regularly spaced. A carefully constructed data type to let the user build regularly-spaced data when working with posteriors would also be very beneficial.
- Interfacing with other packages
    + When [Stheno.jl](https://github.com/willtebbutt/Stheno.jl/) moves over to the AbstractGPs interface, it should be possible to get some interesting process decomposition functionality in this package.
- Approximate inference under non-Gaussian observation models

If you're interested in helping out with this stuff, please get in touch by opening an issue, commenting on an open one, or messaging me on the Julia Slack.



# Relevant literature

See chapter 12 of [1] for the basics.

[1] - Särkkä, Simo, and Arno Solin. Applied stochastic differential equations. Vol. 10. Cambridge University Press, 2019.



# Gotchas

- And time-rescaling is assumed to be a strictly increasing function of time. If this is not the case, then your code will fail silently. Ideally an error would be thrown.
