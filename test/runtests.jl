using BlockDiagonals
using ChainRulesCore
using FillArrays
using FiniteDifferences
using Kronecker
using LinearAlgebra
using Random
using Stheno
using StaticArrays
using StructArrays
using TemporalGPs
using Test
using Zygote

using FiniteDifferences: rand_tangent
using Kronecker: KroneckerProduct
using Stheno: var
using TemporalGPs: AbstractLGSSM, _filter, NoContext
using Zygote: Context, _pullback

if !isdefined(Main, :tests_are_loaded)
    include("test_util.jl")
end

@testset "TemporalGPs.jl" begin

    println("util:")
    @testset "util" begin
    #     include(joinpath("util", "harmonise.jl"))
        # include(joinpath("util", "scan.jl"))
        # include(joinpath("util", "zygote_rules.jl"))
    #     include(joinpath("util", "gaussian.jl"))
    #     include(joinpath("util", "mul.jl"))
    #     include(joinpath("util", "regular_data.jl"))
    end

    println("models:")
    include(joinpath("models", "model_test_utils.jl"))
    # include(joinpath("models", "test_model_test_utils.jl"))
    # @testset "models" begin
    #     include(joinpath("models", "linear_gaussian_conditionals.jl"))
    #     include(joinpath("models", "gauss_markov_model.jl"))
    #     include(joinpath("models", "lgssm.jl"))
    #     # include(joinpath("models", "missings.jl"))
    # end

    println("gp:")
    @testset "gp" begin
        include(joinpath("gp", "lti_sde.jl"))
        # include(joinpath("gp", "posterior_lti_sde.jl"))
    end

    println("space_time:")
    # @testset "space_time" begin
    #     include(joinpath("space_time", "rectilinear_grid.jl"))
    #     include(joinpath("space_time", "separable_kernel.jl"))
    #     include(joinpath("space_time", "to_gauss_markov.jl"))
    # end
end
