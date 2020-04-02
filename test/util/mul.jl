@testset "mul" begin
    rng = MersenneTwister(123456)
    P = 50
    Q = 60
    α = randn(rng)
    β = randn(rng)

    A_Matrix = randn(rng, P, Q)
    At_Matrix = collect(A_Matrix')

    A_blk_diag = BlockDiagonal([randn(rng, P, P), randn(rng, P + 1, P + 1)])
    At_blk_diag = BlockDiagonal(map(collect ∘ transpose, blocks(A_blk_diag)))

    settings = [
        (
            name="Matrix{Float64}",
            A=A_Matrix,
            At=At_Matrix,
            B=randn(rng, size(A_Matrix, 2), Q),
            C=randn(rng, size(A_Matrix, 1), Q),
        ),
        (
            name="BlockDiagonal{Float64, Matrix{Float64}}",
            A=A_blk_diag,
            At=At_blk_diag,
            B=randn(rng, size(A_blk_diag, 2), Q),
            C=randn(rng, size(A_blk_diag, 1), Q),
        ),
        (
            name="BlockDiagonal{Float64, BlockDiagonal{Float64, Matrix{Float64}}}",
            A=BlockDiagonal([A_blk_diag, A_blk_diag]),
            At=BlockDiagonal([At_blk_diag, At_blk_diag]),
            B=randn(rng, 2 * size(A_blk_diag, 2), Q),
            C=randn(rng, 2 * size(A_blk_diag, 1), Q),
        ),
    ]

    @testset "$(setting.name)" for setting in settings
        A = setting.A
        At = setting.At
        B = setting.B
        Bt = collect(B')

        C = setting.C
        Ct = collect(C')

        A_dense = collect(A)
        At_dense = collect(At)

        # Matrix-Matrix product.
        @test collect(mul!(copy(C), A, B, α, β)) ≈ α * A_dense * B + β * C
        @test collect(mul!(copy(C), At', B, α, β)) ≈ α * At_dense' * B + β * C
        @test collect(mul!(copy(C), A, Bt', α, β)) ≈ α * A_dense * Bt' + β * C
        @test collect(mul!(copy(C), At', Bt', α, β)) ≈ α * At_dense' * Bt' + β * C

        @test collect(mul!(copy(Ct), Bt, At, α, β)) ≈ α * Bt * At_dense + β * Ct
        @test collect(mul!(copy(Ct), B', At, α, β)) ≈ α * B' * At_dense + β * Ct
        @test collect(mul!(copy(Ct), Bt, A', α, β)) ≈ α * Bt * A_dense' + β * Ct
        @test collect(mul!(copy(Ct), B', A', α, β)) ≈ α * B' * A_dense' + β * Ct

        # Matrix-Vector product.
        b = B[:, 1]
        c = C[:, 1]
        @test collect(mul!(copy(c), A, b, α, β)) ≈ α * A_dense * b + β * c
    end
end
