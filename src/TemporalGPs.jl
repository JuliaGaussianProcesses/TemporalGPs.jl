module TemporalGPs

    using BlockArrays
    using BlockDiagonals
    using ChainRulesCore
    using Distributions
    using FillArrays
    using Kronecker
    using LinearAlgebra
    using Random
    using StaticArrays
    using Stheno
    using Zygote
    using ZygoteRules

    using FillArrays: AbstractFill
    using Kronecker: KroneckerProduct
    using Zygote: _pullback

    import Stheno:
        mean,
        cov,
        pairwise,
        logpdf,
        AV,
        AM,
        FiniteGP,
        AbstractGP

    export
        to_sde,
        SArrayStorage,
        ArrayStorage,
        RegularSpacing,
        checkpointed,
        posterior,
        logpdf_and_rand

    # Various bits-and-bobs. Often commiting some type piracy.
    include(joinpath("util", "harmonise.jl"))
    include(joinpath("util", "scan.jl"))
    include(joinpath("util", "zygote_rules.jl"))
    include(joinpath("util", "gaussian.jl"))
    include(joinpath("util", "mul.jl"))
    include(joinpath("util", "storage_types.jl"))
    include(joinpath("util", "regular_data.jl"))

    # Linear-Gaussian State Space Models.
    include(joinpath("models", "gauss_markov.jl"))
    include(joinpath("models", "lgssm.jl"))
    include(joinpath("models", "reverse_ssm.jl"))
    include(joinpath("models", "posterior.jl"))
    include(joinpath("models", "scalar_lgssm.jl"))
    # include(joinpath("models", "missings.jl"))

    # Converting GPs to Linear-Gaussian SSMs.
    include(joinpath("gp", "to_gauss_markov.jl"))
    include(joinpath("gp", "lti_sde.jl"))

    include(joinpath("gp", "finite_lti_sde.jl"))
    include(joinpath("gp", "posterior_lti_sde.jl"))

    # Converting space-time GPs to Linear-Gaussian SSMs.
    include(joinpath("space_time", "rectilinear_grid.jl"))
    include(joinpath("space_time", "separable_kernel.jl"))
    include(joinpath("space_time", "to_gauss_markov.jl"))
end # module
