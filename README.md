# TemporalGPs

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://willtebbutt.github.io/TemporalGPs.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://willtebbutt.github.io/TemporalGPs.jl/dev)
[![Build Status](https://travis-ci.com/willtebbutt/TemporalGPs.jl.svg?branch=master)](https://travis-ci.com/willtebbutt/TemporalGPs.jl)
[![Codecov](https://codecov.io/gh/willtebbutt/TemporalGPs.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/willtebbutt/TemporalGPs.jl)

TemporalGPs.jl is a tool to make Gaussian processes (GPs) defined using [Stheno.jl](https://github.com/willtebbutt/Stheno.jl/) fast for time-series. It provides a single-function public API that lets you specify that this package should perform inference, rather than Stheno.jl.



# Example Usage

```julia
using Stheno, TemporalGPs

# Specify a Stheno.jl GP as usual
f_naive = GP(Matern32(), GPC())

# Wrap it in an object that TemporalGPs knows how to handle.
f = to_sde(f_naive)

# Project onto finite-dimensional distribution as usual.
x = range(-5.0, 5.0; length=1000)
fx = f(x, 0.1)

# Sample from the prior as usual.
y = rand(fx_ssm)

# Compute the log marginal likelihood of the data as usual.
logpdf(fx_ssm, y)
```



# Performance Optimisations

There are a couple of ways that `TemporalGPs.jl` can represent things internally. In particular, it can use regular Julia `Vector` and `Matrix` objects, or the `StaticArrays.jl` package to optimise in certain cases. The default is the former. To employ the latter, just add an extra argument to the `ssm` function:
```julia
f = to_sde(f_naive, TemporalGPs.StaticStorage())
```
See the benchmarking results below for the effect that this can have.



# Preliminary Benchmarking Results

![](/examples/preliminary-benchmarks.png)

"naive" timings are with the usual [Stheno.jl](https://github.com/willtebbutt/Stheno.jl/) inference routines, and is the default implementation for GPs. "lgssm" timings are conducted using `ssm` with no additional arguments. "static-lgssm" uses the `TemporalGPs.StaticStorage()` option discussed above.

Gradient computations use Zygote. Custom adjoints have been implemented to achieve this level of performance.



# On-going Work

- Optimisation -- in particular work needs to be done to reduce the allocations made when the default storage is employed.
- Feature coverage -- only a subset of `Stheno.jl`'s functionality is currently available, but it's possible to cover much more.

If you're interested in helping out with this stuff, please get in touch.



# Relevant literature

See chapter 12 of [1] for the basics.

[1] - Särkkä, Simo, and Arno Solin. Applied stochastic differential equations. Vol. 10. Cambridge University Press, 2019.



# Gotchas

- And time-rescaling is assumed to be a strictly increasing function of time. If this is not the case, then your code will fail silently. This could be addressed via careful engineering.
