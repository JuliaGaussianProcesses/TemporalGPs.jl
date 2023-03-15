using StaticArrays
using BenchmarkTools
using BlockDiagonals
using ChainRulesCore
using ChainRulesTestUtils
using Test
using TemporalGPs
using TemporalGPs: time_exp, _map, Gaussian
using FillArrays
using StructArrays
using Zygote: ZygoteRuleConfig
include("../test_util.jl")

@testset "chainrules" begin
    @testset "StaticArrays" begin
        @testset "SArray constructor" begin
            for (f, x) in (
                (SArray{Tuple{3, 2, 1}}, ntuple(i -> 2.5i, 6)),
                (SVector{5}, (ntuple(i -> 2.5i, 5))),
                (SVector{2}, (2.0, 1.0)),
                (SMatrix{5, 4}, (ntuple(i -> 2.5i, 20))),
                (SMatrix{1, 1}, (randn(),))
                )
                test_rrule(ZygoteRuleConfig(), f, x; rrule_f=rrule_via_ad)
            end
        end
        @testset "collect(::SArray)" begin
            A = SArray{Tuple{3, 1, 2}}(ntuple(i -> 3.5i, 6))
            test_rrule(collect, A)
        end
        @testset "vcat(::SVector, ::SVector)" begin
            a = SVector{3}(randn(3))
            b = SVector{2}(randn(2))
            test_rrule(vcat, a, b)
        end
    end
    @testset "time_exp" begin
        A = randn(3, 3)
        test_rrule(time_exp, A ⊢ NoTangent(), 0.1)
    end
    @testset "Fill" begin
        @testset "Fill constructor" begin
            for x in (
                randn(),
                randn(1, 2),
                SMatrix{1, 2}(randn(1, 2)),
            )
                test_rrule(Fill, x, 3; check_inferred=false)
                test_rrule(Fill, x, (3, 4); check_inferred=false)
            end
        end
        @testset "collect(::Fill)" begin
            P = 11
            Q = 3
            @testset "$(typeof(x)) element" for x in [
                randn(),
                randn(1, 2),
                SMatrix{1, 2}(randn(1, 2)),
            ]
                test_rrule(collect, Fill(x, P))
                # The test rule does not work due to inconsistencies of FiniteDifferencies for FillArrays
                test_rrule(collect, Fill(x, P, Q))
            end
        end
    end

    # The rrule is not even used...
    @testset "getindex(::Fill, ::Int)" begin
        X = Fill(randn(5, 3), 10)
        test_rrule(getindex, X, 3; check_inferred=false)
    end
    @testset "BlockDiagonal" begin
        X = map(N -> randn(N, N), [3, 4, 1])
        test_rrule(BlockDiagonal, X)
    end
    @testset "_map(f, x::Fill)" begin
        x = Fill(randn(3, 4), 4)
        test_rrule(_map, sum, x; check_inferred=false)
        test_rrule(_map, x->map(sin, x), x; check_inferred=false)
        test_rrule(_map, x -> 2.0 * x, x; check_inferred=false)
        test_rrule(ZygoteRuleConfig(), (x,a)-> _map(x -> x * a, x), x, 2.0; check_inferred=false, rrule_f=rrule_via_ad)
    end
    @testset "_map(f, x1::Fill, x2::Fill)" begin
        x1 = Fill(randn(3, 4), 3)
        x2 = Fill(randn(3, 4), 3)

        @test _map(+, x1, x2) == _map(+, collect(x1), collect(x2))
        test_rrule(_map, +, x1, x2; check_inferred=true)

        fsin(x, y) = sin.(x .* y)
        test_rrule(_map, fsin, x1, x2; check_inferred=false)

        foo(a, x1, x2) = _map((z1, z2) -> a * sin.(z1 .* z2), x1, x2)
        test_rrule(ZygoteRuleConfig(), foo, randn(), x1, x2; check_inferred=false, rrule_f=rrule_via_ad)
    end
    @testset "StructArray" begin
        a = randn(5)
        b = rand(5)
        # This test is broken due to FiniteDifferences returning the wrong Tangent.
        # test_rrule(StructArray, (a, b); check_inferred=false)

        xs = [Gaussian(randn(1), randn(1, 1)) for _ in 1:2]
        ms = getfield.(xs, :m)
        Ps = getfield.(xs, :P)
        # Same here.
        # test_rrule(StructArray{eltype(xs)}, (ms, Ps))
        xs_sa = StructArray{eltype(xs)}((ms, Ps))
        # And here.
        # test_zygote_grad(getproperty, xs_sa, :m)
    end
end
