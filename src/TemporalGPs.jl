module TemporalGPs

    using FillArrays, LinearAlgebra, Random, StaticArrays, Stheno, Zygote, ZygoteRules

    using FillArrays: AbstractFill

    import Stheno: mean, cov, pairwise, logpdf, AV, AM

    # Used to specify whether to use Base.Array or StaticArray parameter storage.
    abstract type StorageType end
    struct DenseStorage <: StorageType end
    struct StaticStorage <: StorageType end

    # Various bits-and-bobs. Often commiting some type piracy.
    include(joinpath("util", "zygote_rules.jl"))
    include(joinpath("util", "gaussian.jl"))

    # Linear-Gaussian State Space Models.
    include(joinpath("models", "gauss_markov.jl"))
    include(joinpath("models", "lgssm.jl"))
    include(joinpath("models", "lgssm_pullbacks.jl"))
    include(joinpath("models", "scalar_lgssm.jl"))

    # Converting GPs to Linear-Gaussian SSMs.
    include(joinpath("gp", "to_gauss_markov.jl"))
    include(joinpath("gp", "lti_sde.jl"))
end # module
