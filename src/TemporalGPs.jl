module TemporalGPs

using AbstractGPs
using Bessels: besseli
using BlockDiagonals
using FillArrays
using LinearAlgebra
using KernelFunctions
using Random
using StaticArrays
using StructArrays

using FillArrays: AbstractFill

import AbstractGPs: mean, cov, logpdf, FiniteGP, AbstractGP, posterior, dtc, elbo

using KernelFunctions:
    SimpleKernel, KernelSum, ScaleTransform, ScaledKernel, TransformedKernel

export to_sde,
    SArrayStorage, ArrayStorage, RegularSpacing, posterior, Separable, ApproxPeriodicKernel

# Various bits-and-bobs. Often commiting some type piracy.
include(joinpath("util", "linear_algebra.jl"))
include(joinpath("util", "scan.jl"))

include(joinpath("util", "gaussian.jl"))
include(joinpath("util", "mul.jl"))
include(joinpath("util", "storage_types.jl"))
include(joinpath("util", "regular_data.jl"))

# Linear-Gaussian State Space Models.
include(joinpath("models", "linear_gaussian_conditionals.jl"))
include(joinpath("models", "gauss_markov_model.jl"))
include(joinpath("models", "lgssm.jl"))
include(joinpath("models", "missings.jl"))

# Converting GPs to Linear-Gaussian SSMs.
include(joinpath("gp", "data_representations.jl"))
include(joinpath("gp", "lti_sde.jl"))
include(joinpath("gp", "posterior_lti_sde.jl"))

# Converting space-time GPs to Linear-Gaussian SSMs.
include(joinpath("space_time", "rectilinear_grid.jl"))
include(joinpath("space_time", "regular_in_time.jl"))
include(joinpath("space_time", "separable_kernel.jl"))
include(joinpath("space_time", "to_gauss_markov.jl"))
include(joinpath("space_time", "pseudo_point.jl"))
end # module
