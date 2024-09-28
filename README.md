# TemporalGPs

[![CI](https://github.com/JuliaGaussianProcesses/TemporalGPs.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/JuliaGaussianProcesses/TemporalGPs.jl/actions/workflows/ci.yml) [![Coverage Status](https://coveralls.io/repos/github/JuliaGaussianProcesses/TemporalGPs.jl/badge.svg)](https://coveralls.io/github/JuliaGaussianProcesses/TemporalGPs.jl)
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

The following is a small problem by TemporalGPs' standard. See timing results below for expected performance on larger problems.

```julia
using AbstractGPs, KernelFunctions, TemporalGPs

# Specify a AbstractGPs.jl GP as usual
f_naive = GP(Matern32Kernel())

# Wrap it in an object that TemporalGPs knows how to handle.
f = to_sde(f_naive, SArrayStorage(Float64))

# Project onto finite-dimensional distribution as usual.
# x = range(-5.0; step=0.1, length=10_000)
x = RegularSpacing(0.0, 0.1, 10_000) # Hack for AD.
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

## Learning kernel parameters with [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl), [ParameterHandling.jl](https://github.com/invenia/ParameterHandling.jl), and [Mooncake.jl](https://github.com/compintell/Mooncake.jl/)

TemporalGPs.jl doesn't provide scikit-learn-like functionality to train your model (find good kernel parameter settings).
Instead, we offer the functionality needed to easily implement your own training functionality using standard tools from the Julia ecosystem.
See [exact_time_learning.jl](https://github.com/compintell/Mooncake.jl/examples/exact_time_learning.jl).

In this example we optimised the parameters, but we could just as easily have utilised e.g. [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl) in conjunction with a prior over the parameters to perform approximate Bayesian inference in them -- indeed, [this is often a very good idea](http://proceedings.mlr.press/v118/lalchand20a/lalchand20a.pdf).
We leave this as an exercise for the interested user (see e.g. the examples in [Stheno.jl](https://github.com/willtebbutt/Stheno.jl/) for inspiration).

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

Gradient computations use Mooncake. Custom adjoints have been implemented to achieve this level of performance.


# Relevant literature

See chapter 12 of [1] for the basics.

[1] - Särkkä, Simo, and Arno Solin. Applied stochastic differential equations. Vol. 10. Cambridge University Press, 2019.



# Gotchas

- And time-rescaling is assumed to be a strictly increasing function of time. If this is not the case, then your code will fail silently. Ideally an error would be thrown.
