using StaticArrays
using TemporalGPs: time_exp, logdet_pullback

@testset "zygote_rules" begin
    @testset "SVector" begin
        adjoint_test(SVector{5}, (randn(5), ))
    end
    @testset "SMatrix" begin
        adjoint_test(SMatrix{5, 4}, (randn(5, 4), ))
    end
    @testset "SMatrix{1, 1} from scalar" begin
        adjoint_test(SMatrix{1, 1}, (randn(), ))
    end
    @testset "time_exp" begin
        A = randn(3, 3)
        adjoint_test(t->time_exp(A, t), (0.1, ))
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
    # @testset "reinterpret" begin
    #     P = 11
    #     T = SVector{1, Float64}
    #     adjoint_test(y->reinterpret(T, y), (randn(11), ); check_infers=false)
    #     adjoint_test(α->reinterpret(Float64, α), (T.(randn(5)), ))
    # end
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

        @test map(+, x1, x2) == map(+, collect(x1), collect(x2))
        adjoint_test((x1, x2) -> map(+, x1, x2), (x1, x2))

        adjoint_test(
            (x1, x2) -> map((z1, z2) -> sin.(z1 .* z2), x1, x2), (x1, x2);
            check_infers=false,
        )

        foo = (a, x1, x2) -> begin
            return map((z1, z2) -> a * sin.(z1 .* z2), x1, x2)
        end
        adjoint_test(foo, (randn(), x1, x2); check_infers=false)
    end
    @testset "$N, $T" for N in [1, 2, 3], T in [Float32, Float64]

        rng = MersenneTwister(123456)

        # Do dense stuff.
        S_ = randn(rng, T, N, N)
        S = S_ * S_' + I
        C = cholesky(S)
        Ss = SMatrix{N, N, T}(S)
        Cs = cholesky(Ss)

        @testset "cholesky" begin
            C_fwd, pb = Zygote.pullback(cholesky, Symmetric(S))
            Cs_fwd, pbs = Zygote.pullback(cholesky, Symmetric(Ss))

            @test eltype(C_fwd) == T
            @test eltype(Cs_fwd) == T

            ΔC = randn(rng, T, N, N)
            ΔCs = SMatrix{N, N, T}(ΔC)

            @test C.U ≈ Cs.U
            @test Cs.U ≈ Cs_fwd.U

            ΔS, = pb((factors=ΔC, ))
            ΔSs, = pbs((factors=ΔCs, ))

            @test ΔS ≈ ΔSs.data
            @test eltype(ΔS) == T
            @test eltype(ΔSs.data) == T

            @test allocs(@benchmark(cholesky(Symmetric($Ss)); samples=1, evals=1)) == 0
            @test allocs(@benchmark(Zygote._pullback($(Context()), cholesky, Symmetric($Ss)); samples=1, evals=1)) == 0
            @test allocs(@benchmark($pbs((factors=$ΔCs,)); samples=1, evals=1)) == 0
        end
        @testset "logdet" begin
            @test logdet(Cs) ≈ logdet(C)
            C_fwd, pb = logdet_pullback(C)
            Cs_fwd, pbs = logdet_pullback(Cs)

            @test eltype(C_fwd) == T
            @test eltype(Cs_fwd) == T

            @test logdet(Cs) ≈ Cs_fwd

            Δ = randn(rng, T)
            ΔC = first(pb(Δ)).factors
            ΔCs = first(pbs(Δ)).factors

            @test ΔC ≈ ΔCs
            @test eltype(ΔC) == T
            @test eltype(ΔCs) == T

            @test allocs(@benchmark(logdet($Cs); samples=1, evals=1)) == 0
            @test allocs(@benchmark(logdet_pullback($Cs); samples=1, evals=1)) == 0
            @test allocs(@benchmark($pbs($Δ); samples=1, evals=1)) == 0
        end
    end
end
