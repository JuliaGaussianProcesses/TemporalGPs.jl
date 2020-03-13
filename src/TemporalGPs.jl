module TemporalGPs

    using FillArrays, LinearAlgebra, Random, StaticArrays, Stheno, Zygote, ZygoteRules

    import Stheno: pairwise, logpdf, AV, AM

    # Various bits-and-bobs. Often involved in some type piracy.
    include(joinpath("util", "zygote_rules.jl"))
    include(joinpath("util", "gaussian.jl"))

    # Converting GPs to State Space Models. Includes (orthogonal) spatio-temporal models.
    include("gp_to_sde.jl")

    # Linear-Gaussian State Space Models.
    include(joinpath("lgssm", "generic.jl"))
    include(joinpath("lgssm", "generic_pullbacks.jl"))
    include(joinpath("lgssm", "scalar.jl"))
end # module
