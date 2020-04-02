@testset "mul" begin
    @testset "Matrix" begin
        rng = MersenneTwister(123456)
        P = 5
        Q = 4
        R = 6
        A = randn(rng, P, Q)
        At = collect(A')
        B = randn(rng, Q, R)
        Bt = collect(B')
        C = randn(rng, P, R)
        Ct = collect(C')
        α = randn(rng)
        β = randn(rng)

        @test mul!(copy(C), A, B, α, β) ≈ α * A * B + β * C
        @test mul!(copy(C), At', B, α, β) ≈ α * At' * B + β * C
        @test mul!(copy(C), A, Bt', α, β) ≈ α * A * Bt' + β * C
        @test mul!(copy(C), At', Bt', α, β) ≈ α * At' * Bt' + β * C

        @test mul!(copy(Ct), Bt, At, α, β) ≈ α * Bt * At + β * Ct
        @test mul!(copy(Ct), B', At, α, β) ≈ α * B' * At + β * Ct
        @test mul!(copy(Ct), Bt, A', α, β) ≈ α * Bt * A' + β * Ct
        @test mul!(copy(Ct), B', A', α, β) ≈ α * B' * A' + β * Ct
    end
end
