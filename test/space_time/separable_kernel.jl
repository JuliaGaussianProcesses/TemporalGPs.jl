using Random
using TemporalGPs: RectilinearGrid, Separable

@testset "separable_kernel" begin
    rng = MersenneTwister(123456)

    k = Separable(EQ(), Matern32())
    x0 = collect(RectilinearGrid(randn(rng, 2), randn(rng, 3)))
    x1 = collect(RectilinearGrid(randn(rng, 2), randn(rng, 3)))
    x2 = collect(RectilinearGrid(randn(rng, 3), randn(rng, 1)))
    atol=1e-9

    # Check that elementwise basically works.
    @test ew(k, x0, x1) isa AbstractVector
    @test length(ew(k, x0, x1)) == length(x0)

    # Check that pairwise basically works.
    @test pw(k, x0, x2) isa AbstractMatrix
    @test size(pw(k, x0, x2)) == (length(x0), length(x2))

    # Check that elementwise is consistent with pairwise.
    @test ew(k, x0, x1) ≈ diag(pw(k, x0, x1)) atol=atol

    # Check additional binary elementwise properties for kernels.
    @test ew(k, x0, x1) ≈ ew(k, x1, x0)
    @test pw(k, x0, x2) ≈ pw(k, x2, x0)' atol=atol

    # Check that unary elementwise basically works.
    @test ew(k, x0) isa AbstractVector
    @test length(ew(k, x0)) == length(x0)

    # Check that unary pairwise basically works.
    @test pw(k, x0) isa AbstractMatrix
    @test size(pw(k, x0)) == (length(x0), length(x0))
    @test pw(k, x0) ≈ pw(k, x0)' atol=atol

    # Check that unary elementwise is consistent with unary pairwise.
    @test ew(k, x0) ≈ diag(pw(k, x0)) atol=atol

    # Check that unary pairwise produces a positive definite matrix (approximately).
    @test all(eigvals(Matrix(pw(k, x0))) .> -atol)

    # Check that unary elementwise / pairwise are consistent with the binary versions.
    @test ew(k, x0) ≈ ew(k, x0, x0) atol=atol
    @test pw(k, x0) ≈ pw(k, x0, x0) atol=atol
end
