using BlockDiagonals
using Documenter
using FillArrays
using FiniteDifferences
using Kronecker
using LinearAlgebra
using Random
using Stheno
using StaticArrays
using TemporalGPs
using Test
using Zygote

using Kronecker: KroneckerProduct

include("test_util.jl")

@testset "TemporalGPs.jl" begin

    # println("util:")
    # @testset "util" begin
    #     include(joinpath("util", "zygote_rules.jl"))
    #     include(joinpath("util", "gaussian.jl"))
    #     include(joinpath("util", "mul.jl"))
    #     include(joinpath("util", "regular_data.jl"))
    # end

    include(joinpath("models", "model_test_utils.jl"))
    include(joinpath("models", "test_model_test_utils.jl"))
    # @testset "models" begin
    #     include(joinpath("models", "gauss_markov.jl"))
    #     include(joinpath("models", "lgssm.jl"))

    #     include(joinpath("models", "immutable_inference.jl"))
    #     include(joinpath("models", "immutable_inference_pullbacks.jl"))
    #     include(joinpath("models", "checkpointed_immutable_pullbacks.jl"))

    #     include(joinpath("models", "scalar_lgssm.jl"))
    # end

    @testset "gp" begin
        # include(joinpath("gp", "to_gauss_markov.jl"))
        # include(joinpath("gp", "lti_sde.jl"))
        # include(joinpath("gp", "finite_lti_sde.jl"))
        include(joinpath("gp", "posterior_lti_sde.jl"))
    end

    # println("space_time:")
    # @testset "space_time" begin
    #     include(joinpath("space_time", "rectilinear_grid.jl"))
    #     include(joinpath("space_time", "separable_kernel.jl"))
    #     include(joinpath("space_time", "to_gauss_markov.jl"))
    # end

    println("doctests")
    @testset "doctests" begin
        DocMeta.setdocmeta!(
            TemporalGPs,
            :DocTestSetup,
            :(using Documenter, LinearAlgebra, Random, TemporalGPs);
            recursive=true,
        )
        doctest(TemporalGPs)
    end
end
