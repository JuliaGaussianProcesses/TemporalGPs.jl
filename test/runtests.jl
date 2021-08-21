using Test

ENV["TESTING"] = "TRUE"
const GROUP = get(ENV, "GROUP", "test")

OUTER_GROUP = first(split(GROUP, ' '))

# Run the tests.
if OUTER_GROUP == "test" || OUTER_GROUP == "all"

    # Determines which group of tests should be run.
    group_info = split(GROUP, ' ')
    TEST_GROUP = length(group_info) == 1 ? "all" : group_info[2]

    using AbstractGPs
    using BlockDiagonals
    using ChainRulesCore
    using FillArrays
    using FiniteDifferences
    using LinearAlgebra
    using KernelFunctions
    using Random
    using StaticArrays
    using StructArrays
    using TemporalGPs

    using Zygote

    using FiniteDifferences: rand_tangent
    using AbstractGPs: var
    using TemporalGPs: AbstractLGSSM, _filter, NoContext
    using Zygote: Context, _pullback

    include("test_util.jl")

    @show TEST_GROUP GROUP

    @testset "TemporalGPs.jl" begin

        if TEST_GROUP == "util" || GROUP == "all"
            println("util:")
            @testset "util" begin
                include(joinpath("util", "harmonise.jl"))
                include(joinpath("util", "scan.jl"))
                include(joinpath("util", "zygote_friendly_map.jl"))
                include(joinpath("util", "zygote_rules.jl"))
                include(joinpath("util", "gaussian.jl"))
                include(joinpath("util", "mul.jl"))
                include(joinpath("util", "regular_data.jl"))
            end
        end

        if TEST_GROUP ∈ ["models", "models-lgssm", "gp", "space_time"] || GROUP == "all"
            include(joinpath("models", "model_test_utils.jl"))
            include(joinpath("models", "test_model_test_utils.jl"))
        end

        if TEST_GROUP == "models" || GROUP == "all"
            println("models:")
            @testset "models" begin
                include(joinpath("models", "linear_gaussian_conditionals.jl"))
                include(joinpath("models", "gauss_markov_model.jl"))
                include(joinpath("models", "missings.jl"))
            end
        end

        if TEST_GROUP == "models-lgssm" || GROUP == "all"
            println("models (lgssm):")
            @testset "models (lgssm)" begin
                include(joinpath("models", "lgssm.jl"))
            end
        end

        if TEST_GROUP == "gp" || GROUP == "all"
            println("gp:")
            @testset "gp" begin
                include(joinpath("gp", "lti_sde.jl"))
                include(joinpath("gp", "posterior_lti_sde.jl"))
            end
        end

        if TEST_GROUP == "space_time" || GROUP == "all"
            println("space_time:")
            @testset "space_time" begin
                include(joinpath("space_time", "rectilinear_grid.jl"))
                include(joinpath("space_time", "regular_in_time.jl"))
                include(joinpath("space_time", "separable_kernel.jl"))
                include(joinpath("space_time", "to_gauss_markov.jl"))
                include(joinpath("space_time", "pseudo_point.jl"))
            end
        end
    end
end



# Run the examples.
if GROUP == "examples"

    using Pkg

    Pkg.activate(joinpath("..", "examples"))
    Pkg.develop(path="..")
    Pkg.resolve()
    Pkg.instantiate()

    include(joinpath("..", "examples", "exact_time_inference.jl"))
    include(joinpath("..", "examples", "exact_time_learning.jl"))
    include(joinpath("..", "examples", "exact_space_time_inference.jl"))
    include(joinpath("..", "examples", "exact_space_time_learning.jl"))
    include(joinpath("..", "examples", "approx_space_time_inference.jl"))
    include(joinpath("..", "examples", "approx_space_time_learning.jl"))
    include(joinpath("..", "examples", "augmented_inference.jl"))
end
