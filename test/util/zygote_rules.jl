using StaticArrays
using TemporalGPs: time_exp

@testset "zygote_rules" begin
    @testset "SVector" begin
        rng = MersenneTwister(123456)
        N = 5
        x = randn(rng, N)
        ȳ = SVector{N}(randn(rng, N))
        adjoint_test(SVector{N}, ȳ, x)
    end
    @testset "SMatrix" begin
        rng = MersenneTwister(123456)
        P, Q = 5, 4
        X = randn(rng, P, Q)
        Ȳ = SMatrix{P, Q}(randn(rng, P, Q))
        adjoint_test(SMatrix{P, Q}, Ȳ, X)
    end
    @testset "SMatrix{1, 1} from scalar" begin
        rng = MersenneTwister(123456)
        x = randn(rng)
        ȳ = SMatrix{1, 1}(randn(rng))
        adjoint_test(SMatrix{1, 1}, ȳ, x)
    end
    @testset "time_exp" begin
        rng = MersenneTwister(123456)
        A = randn(rng, 3, 3)
        t = 0.1
        ΔB = randn(rng, 3, 3)
        adjoint_test(t->time_exp(A, t), ΔB, t)
    end
    @testset "collect(::Fill)" begin
        rng = MersenneTwister(123456)
        P = 11
        Q = 3
        xs = [
            randn(rng),
            randn(rng, 1, 2),
            SMatrix{1, 2}(randn(rng, 1, 2)),
        ]
        Δs = [
            (randn(rng, P), randn(rng, P, Q)),
            ([randn(rng, 1, 2) for _ in 1:P], [randn(rng, 1, 2) for _ in 1:P, _ in 1:Q]),
            (
                [SMatrix{1, 2}(randn(rng, 1, 2)) for _ in 1:P],
                [SMatrix{1, 2}(randn(rng, 1, 2)) for _ in 1:P, _ in 1:Q],
            ),
        ]
        @testset "$(typeof(x)) element" for (x, Δ) in zip(xs, Δs)
            adjoint_test(x->collect(Fill(x, P)), first(Δ), x)
            adjoint_test(x->collect(Fill(x, P, Q)), last(Δ), x)
        end
    end
    @testset "reinterpret" begin
        rng = MersenneTwister(123456)
        P = 11
        y = randn(rng, P)
        Δy = randn(rng, P)
        T = SVector{1, Float64}
        α = T.(randn(rng, P))
        Δα = T.(randn(rng, P))
        adjoint_test(y->reinterpret(T, y), Δα, y)
        adjoint_test(α->reinterpret(Float64, α), Δy, α)
    end
    @testset "getindex(::Fill, ::Int)" begin
        N = 11
        val = randn(5, 3)
        adjoint_test(val -> getindex(Fill(val, N), 3), randn(size(val)), val)
    end
    @testset "BlockDiagonal" begin
        rng = MersenneTwister(123456)
        Ns = [3, 4, 1]
        Xs = map(N -> randn(rng, N, N), Ns)
        ΔX = (blocks=map(N -> randn(rng, N, N), Ns),)
        adjoint_test(BlockDiagonal, ΔX, Xs)
    end
    @testset "map(f, x::Fill)" begin
        rng = MersenneTwister(123456)
        N = 5
        x = Fill(randn(rng, 3, 4), 4)
        ȳ = (value = randn(rng),)
        adjoint_test(x -> map(sum, x), ȳ, x)

        ȳ = (value = randn(rng, 3, 4),)
        adjoint_test(x -> map(x -> map(z -> sin(z), x), x), ȳ, x)

        foo = (a, x) -> begin
            return map(x -> a * x, x)
        end
        adjoint_test(foo, ȳ, randn(rng), x)
    end
    @testset "map(f, x1::Fill, x2::Fill)" begin
        rng = MersenneTwister(123456)
        N = 5
        x1 = Fill(randn(rng, 3, 4), 3)
        x2 = Fill(randn(rng, 3, 4), 3)
        ȳ = (value = randn(rng, 3, 4),)

        @test map(+, x1, x2) == map(+, collect(x1), collect(x2))
        adjoint_test((x1, x2) -> map(+, x1, x2), ȳ, x1, x2)

        adjoint_test((x1, x2) -> map((z1, z2) -> sin.(z1 .* z2), x1, x2), ȳ, x1, x2)

        foo = (a, x1, x2) -> begin
            return map((z1, z2) -> a * sin.(z1 .* z2), x1, x2)
        end
        adjoint_test(foo, ȳ, randn(rng), x1, x2)
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
            C_fwd, pb = cholesky_pullback(Symmetric(S))
            Cs_fwd, pbs = cholesky_pullback(Symmetric(Ss))

            @test eltype(C_fwd) == T
            @test eltype(Cs_fwd) == T

            ΔC = randn(rng, T, N, N)
            ΔCs = SMatrix{N, N, T}(ΔC)

            @test C.U ≈ Cs.U
            @test Cs.U ≈ Cs_fwd.U

            ΔS, = pb((factors=ΔC, ))
            ΔSs, = pbs((factors=ΔCs, ))

            @test ΔS ≈ ΔSs
            @test eltype(ΔS) == T
            @test eltype(ΔSs) == T

            @test allocs(@benchmark cholesky(Symmetric($Ss))) == 0
            @test allocs(@benchmark cholesky_pullback(Symmetric($Ss))) == 0
            @test allocs(@benchmark $pbs((factors=$ΔCs,))) == 0
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

            @test allocs(@benchmark logdet($Cs)) == 0
            @test allocs(@benchmark logdet_pullback($Cs)) == 0
            @test allocs(@benchmark $pbs($Δ)) == 0
        end
    end
end
