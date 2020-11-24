# TemporalGPs

[![Build Status](https://github.com/willtebbutt/TemporalGPs.jl/workflows/CI/badge.svg)](https://github.com/willtebbutt/TemporalGPs.jl/actions)
[![Codecov](https://codecov.io/gh/willtebbutt/TemporalGPs.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/willtebbutt/TemporalGPs.jl)

TemporalGPs.jl is a tool to make Gaussian processes (GPs) defined using [Stheno.jl](https://github.com/willtebbutt/Stheno.jl/) fast for time-series. It provides a single-function public API that lets you specify that this package should perform inference, rather than Stheno.jl.

# Installation

TemporalGPs.jl is registered, so simply type the following at the REPL:
```julia
] add Stheno TemporalGPs
```
While you can install TemporalGPs without Stheno, in practice the latter is needed for all common tasks in TemporalGPs.

# Example Usage

This is a small problem by TemporalGPs' standard. See timing results below for expected performance on larger problems.

```julia
using Stheno, TemporalGPs

# Specify a Stheno.jl GP as usual
f_naive = GP(Matern32(), GPC())

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



# Performance Optimisations

There are a couple of ways that `TemporalGPs.jl` can represent things internally. In particular, it can use regular Julia `Vector` and `Matrix` objects, or the `StaticArrays.jl` package to optimise in certain cases. The default is the former. To employ the latter, just add an extra argument to the `to_sde` function:
```julia
f = to_sde(f_naive, SArrayStorage(Float64))
```
This tells TemporalGPs that you want all parameters of `f` and anything derived from it to be a subtype of a `SArray` with element-type `Float64`, rather than (for example) a `Matrix{Float64}`s and `Vector{Float64}`. The decision made here can have quite a dramatic effect on performance, as shown in the graph below. For "larger" kernels (large sums, spatio-temporal problems), you might want to consider `ArrayStorage(Float64)` instead.



# Benchmarking Results

![](/examples/benchmarks.png)

"naive" timings are with the usual [Stheno.jl](https://github.com/willtebbutt/Stheno.jl/) inference routines, and is the default implementation for GPs. "lgssm" timings are conducted using `to_sde` with no additional arguments. "static-lgssm" uses the `SArrayStorage(Float64)` option discussed above.

Gradient computations use Zygote. Custom adjoints have been implemented to achieve this level of performance.



# On-going Work

- Optimisation
    + in-place implementation with `ArrayStorage` to reduce allocations
    + input data types for posterior inference - the `RegularSpacing` type is great for expressing that the inputs are regularly spaced. A carefully constructed data type to let the user build regularly-spaced data when working with posteriors would also be very beneficial.
- Feature coverage
    + only a subset of `Stheno.jl`'s probabilistic-programming functionality is currently available, but it's possible to cover much more.
    + reverse-mode through posterior inference. This is quite straightforward in principle, it just requires a couple of extra ChainRules.
- Interfacing with other packages
    + Both Stheno and this package will move over to the AbstractGPs.jl interface at some point, which will enable both to interface more smoothly with other packages in the ecosystem.

If you're interested in helping out with this stuff, please get in touch by opening an issue, commenting on an open one, or messaging me on the Julia Slack.



# Relevant literature

See chapter 12 of [1] for the basics.

[1] - Särkkä, Simo, and Arno Solin. Applied stochastic differential equations. Vol. 10. Cambridge University Press, 2019.



# Gotchas

- And time-rescaling is assumed to be a strictly increasing function of time. If this is not the case, then your code will fail silently. Ideally an error would be thrown.
