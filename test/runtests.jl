include("front_matter.jl")

# Run the tests.
@testset "TemporalGPs.jl" begin
    if GROUP == "quality"
        Aqua.test_all(TemporalGPs)
        @test JuliaFormatter.format(TemporalGPs; verbose=false, overwrite=false)
    end

    if GROUP == "test util"
        println("util:")
        @testset "util" begin
            include(joinpath("util", "scan.jl"))
            include(joinpath("util", "gaussian.jl"))
            include(joinpath("util", "mul.jl"))
            include(joinpath("util", "regular_data.jl"))
        end
    end

    if GROUP == "test models"
        @testset "models" begin
            println("models:")
            include(joinpath("models", "test_model_test_utils.jl"))
            include(joinpath("models", "linear_gaussian_conditionals.jl"))
            include(joinpath("models", "gauss_markov_model.jl"))
            include(joinpath("models", "lgssm.jl"))
            include(joinpath("models", "missings.jl"))
        end
    end

    if GROUP == "test gp"
        println("gp:")
        @testset "gp" begin
            include(joinpath("gp", "lti_sde.jl"))
            include(joinpath("gp", "posterior_lti_sde.jl"))
        end
    end

    if GROUP == "test space_time"
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

# Run the examples.
if GROUP == "examples"
    using Pkg
    pkgpath = joinpath(@__DIR__, "..")
    Pkg.activate(joinpath(pkgpath, "examples"))
    Pkg.develop(; path=pkgpath)
    Pkg.resolve()
    Pkg.instantiate()

    function include_with_info(filename)
        @info "Running examples/$filename"
        return include(joinpath(pkgpath, "examples", filename))
    end

    include_with_info("exact_time_inference.jl")
    include_with_info("exact_time_learning.jl")
    include_with_info("exact_space_time_inference.jl")
    include_with_info("exact_space_time_learning.jl")
    include_with_info("approx_space_time_inference.jl")
    include_with_info("approx_space_time_learning.jl")
    include_with_info("augmented_inference.jl")
end
