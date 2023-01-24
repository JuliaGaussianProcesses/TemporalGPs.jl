using StaticArrays
using BenchmarkTools
using ChainRulesCore
using ChainRulesTestUtils
using Test
using TemporalGPs
using TemporalGPs: time_exp, _map
using FillArrays
using Zygote: ZygoteRuleConfig

@testset "Test rrules" begin
    @testset "SArray" begin
        # test_rrule()
    end

    @testset "_map" begin
        σ = 2.0
        # test_rrule(TemporalGPs._scale_emission_projections, ([Fill(1.0, 10) for _ in 1:2], [Fill(2.0, 10)] for _ in 1:2), 2.0)
        N = 2
        tgt = Tangent{Tuple}(ntuple(_ -> Tangent{Any}(NoTangent(), [Tangent{Fill}(value=1.0, axes=NoTangent())]), N))
        test_rrule(ZygoteRuleConfig(), TemporalGPs._map ⊢ tgt, x -> σ * x, ([Fill(1.0, 10) for _ in 1:N], [Fill(2.0, 10) for _ in 1:N]); rrule_f=rrule_via_ad, check_inferred=false)
    end
end

@testset "chainrules" begin
    @testset "SArray" begin
        for (f, x) in (
            (SArray{Tuple{3, 2, 1}}, ntuple(i -> 2.5i, 6)),
            (SVector{5}, (ntuple(i -> 2.5i, 5))),
            (SVector{2}, (2.0, 1.0)),
            (SMatrix{5, 4}, (ntuple(i -> 2.5i, 20))),
            (SMatrix{1, 1}, (randn(),))
            )
            test_rrule(ZygoteRuleConfig(), f, x; rrule_f=rrule_via_ad, check_inferred=false)
        end
    end
    # adjoint_test(SArray{Tuple{3, 2, 1}}, (ntuple(i -> 2.5i, 6), ))
    # _, pb = Zygote._pullback(SArray{Tuple{3, 2, 1}}, ntuple(i -> 2.5i, 6))
    # pb(nothing) === (nothing, nothing)
    # @testset "SVector" begin
        # adjoint_test(SVector{5}, (ntuple(i -> 2.5i, 5), ))
        # adjoint_test(SVector{2}, (2.0, 1.0))
    # end
    # @testset "SMatrix" begin
        # adjoint_test(SMatrix{5, 4}, (ntuple(i -> 2.5i, 20), ))
    # end
    # @testset "SMatrix{1, 1} from scalar" begin
        # adjoint_test(SMatrix{1, 1}, (randn(), ))
    # end
    @testset "time_exp" begin
        A = randn(3, 3)
        test_rrule(time_exp, A ⊢ NoTangent(), 0.1)
    end
    @testset "collect(::SArray)" begin
        A = SArray{Tuple{3, 1, 2}}(ntuple(i -> 3.5i, 6))
        # test_rrule(collect, A)
        adjoint_test(collect, (A, ))
    end
    @testset "vcat(::SVector, ::SVector)" begin
        a = SVector{3}(randn(3))
        b = SVector{2}(randn(2))
        adjoint_test(vcat, (a, b))
    end
    @testset "collect(::Fill)" begin
        P = 11
        Q = 3
        @testset "$(typeof(x)) element" for x in [
            randn(),
            randn(1, 2),
            SMatrix{1, 2}(randn(1, 2)),
        ]
            adjoint_test(collect, (Fill(x, P), ))
            adjoint_test(collect, (Fill(x, P, Q), ))
        end
    end
    @testset "getindex(::Fill, ::Int)" begin
        adjoint_test(x -> getindex(x, 3), (Fill(randn(5, 3), 10),))
    end
    @testset "BlockDiagonal" begin
        adjoint_test(BlockDiagonal, (map(N -> randn(N, N), [3, 4, 1]), ))
    end
    @testset "map(f, x::Fill)" begin
        x = Fill(randn(3, 4), 4)
        adjoint_test(x -> map(sum, x), (x, ))
        adjoint_test(x -> map(x -> map(z -> sin(z), x), x), (x, ); check_infers=false)
        adjoint_test((a, x) -> map(x -> a * x, x), (randn(), x))
    end
    @testset "map(f, x1::Fill, x2::Fill)" begin
        x1 = Fill(randn(3, 4), 3)
        x2 = Fill(randn(3, 4), 3)

        @test _map(+, x1, x2) == _map(+, collect(x1), collect(x2))
        test_rrule(ZygoteRuleConfig(), _map, +, x1, x2; rrule_f=rrule_via_ad, check_inferred=false)
        adjoint_test((x1, x2) -> _map(+, x1, x2), (x1, x2))

        adjoint_test(
            (x1, x2) -> map((z1, z2) -> sin.(z1 .* z2), x1, x2), (x1, x2);
            check_infers=false,
        )

        foo = (a, x1, x2) -> begin
            return map((z1, z2) -> a * sin.(z1 .* z2), x1, x2)
        end
        adjoint_test(foo, (randn(), x1, x2); check_infers=false)
    end
    # @testset "$N, $T" for N in [1, 2, 3], T in [Float32, Float64]

    #     rng = MersenneTwister(123456)

    #     # Do dense stuff.
    #     S_ = randn(rng, T, N, N)
    #     S = S_ * S_' + I
    #     C = cholesky(S)
    #     Ss = SMatrix{N, N, T}(S)
    #     Cs = cholesky(Ss)

    #     @testset "cholesky" begin
    #         C_fwd, pb = Zygote.pullback(cholesky, Symmetric(S))
    #         Cs_fwd, pbs = Zygote.pullback(cholesky, Symmetric(Ss))

    #         @test eltype(C_fwd) == T
    #         @test eltype(Cs_fwd) == T

    #         ΔC = randn(rng, T, N, N)
    #         ΔCs = SMatrix{N, N, T}(ΔC)

    #         @test C.U ≈ Cs.U
    #         @test Cs.U ≈ Cs_fwd.U

    #         ΔS, = pb((factors=ΔC, ))
    #         ΔSs, = pbs((factors=ΔCs, ))

    #         @test ΔS ≈ ΔSs.data
    #         @test eltype(ΔS) == T
    #         @test eltype(ΔSs.data) == T

    #         @test allocs(@benchmark(cholesky(Symmetric($Ss)); samples=1, evals=1)) == 0
    #         @test allocs(@benchmark(Zygote._pullback($(Context()), cholesky, Symmetric($Ss)); samples=1, evals=1)) == 0
    #         @test allocs(@benchmark($pbs((factors=$ΔCs,)); samples=1, evals=1)) == 0
    #     end
    #     @testset "logdet" begin
    #         @test logdet(Cs) ≈ logdet(C)
    #         C_fwd, pb = logdet_pullback(C)
    #         Cs_fwd, pbs = logdet_pullback(Cs)

    #         @test eltype(C_fwd) == T
    #         @test eltype(Cs_fwd) == T

    #         @test logdet(Cs) ≈ Cs_fwd

    #         Δ = randn(rng, T)
    #         ΔC = first(pb(Δ)).factors
    #         ΔCs = first(pbs(Δ)).factors

    #         @test ΔC ≈ ΔCs
    #         @test eltype(ΔC) == T
    #         @test eltype(ΔCs) == T

    #         @test allocs(@benchmark(logdet($Cs); samples=1, evals=1)) == 0
    #         @test allocs(@benchmark(logdet_pullback($Cs); samples=1, evals=1)) == 0
    #         @test allocs(@benchmark($pbs($Δ); samples=1, evals=1)) == 0
    #     end
    # end
    @testset "StructArray" begin
        a = randn(5)
        b = rand(5)
        adjoint_test(StructArray, ((a, b), ))
        # adjoint_test(StructArray, ((a=a, b=b), ))

        xs = [Gaussian(randn(1), randn(1, 1)) for _ in 1:2]
        ms = getfield.(xs, :m)
        Ps = getfield.(xs, :P)
        adjoint_test(StructArray{eltype(xs)}, ((ms, Ps), ))

        xs_sa = StructArray{eltype(xs)}((ms, Ps))
        adjoint_test(xs -> xs.m, (xs_sa, ))
    end
    @testset "\\" begin
        adjoint_test(\, (Diagonal(rand(5) .+ 1.0), randn(5)))
        adjoint_test(\, (Diagonal(rand(5) .+ 1.0), randn(5, 2)))
    end
    @testset ".\\" begin
        adjoint_test((a, x) -> a .\ x, (randn(10), randn(10)); rtol=1e-7, atol=1e-7)
        adjoint_test((a, x) -> a .\ x, (randn(10), randn(10, 3)); rtol=1e-7, atol=1e-7)
        adjoint_test((a, x) -> a .\ x, (randn(3), randn(3, 10)); rtol=1e-7, atol=1e-7)
    end
end
