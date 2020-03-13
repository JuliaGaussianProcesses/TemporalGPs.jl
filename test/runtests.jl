using FiniteDifferences, TemporalGPs, Test
using FillArrays, LinearAlgebra, Random, Stheno, StaticArrays, Zygote

include("test_util.jl")

@testset "TemporalGPs.jl" begin
    @testset "util" begin
        include(joinpath("util", "zygote_rules.jl"))
        include(joinpath("util", "gaussian.jl"))
    end

    include("gp_to_sde.jl")

    @testset "lgssm" begin
        include(joinpath("lgssm", "generic.jl"))
        include(joinpath("lgssm", "scalar.jl"))
    end
end
