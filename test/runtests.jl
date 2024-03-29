using Test

ENV["TESTING"] = "TRUE"

# GROUP is an env variable from CI which can take the following values
# ["test util", "test models" "test models-lgssm" "test gp" "test space_time"]
# Select any of this to test a particular aspect.
# To test everything, simply set GROUP to "all"
# ENV["GROUP"] = "test gp"
const GROUP = get(ENV, "GROUP", "test")
OUTER_GROUP = first(split(GROUP, ' '))

const TEST_TYPE_INFER = false # Test type stability over the tests
const TEST_ALLOC = false # Test allocations over the tests

# Run the tests.
if OUTER_GROUP == "test" || OUTER_GROUP == "all"

    # Determines which group of tests should be run.
    group_info = split(GROUP, ' ')
    TEST_GROUP = length(group_info) == 1 ? "all" : group_info[2]

    using AbstractGPs
    using BlockDiagonals
    using ChainRulesCore
    using ChainRulesTestUtils
    using FillArrays
    using FiniteDifferences
    using LinearAlgebra
    using KernelFunctions
    using Random
    using StaticArrays
    using StructArrays
    using TemporalGPs

    using Zygote

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
                include(joinpath("util", "chainrules.jl"))
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
    pkgpath = joinpath(@__DIR__, "..")
    Pkg.activate(joinpath(pkgpath, "examples"))
    Pkg.develop(path=pkgpath)
    Pkg.resolve()
    Pkg.instantiate()

    function include_with_info(filename)
        @info "Running examples/$filename"
        include(joinpath(pkgpath, "examples", filename))
    end

    include_with_info("exact_time_inference.jl")
    include_with_info("exact_time_learning.jl")
    include_with_info("exact_space_time_inference.jl")
    include_with_info("exact_space_time_learning.jl")
    include_with_info("approx_space_time_inference.jl")
    include_with_info("approx_space_time_learning.jl")
    include_with_info("augmented_inference.jl")
end
