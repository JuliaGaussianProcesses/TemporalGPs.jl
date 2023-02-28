using Random
using TemporalGPs: RectilinearGrid, Separable

@testset "separable_kernel" begin
    rng = MersenneTwister(123456)

    k = Separable(SEKernel(), Matern32Kernel())
    x0 = collect(RectilinearGrid(randn(rng, 2), randn(rng, 3)))
    x1 = collect(RectilinearGrid(randn(rng, 2), randn(rng, 3)))
    x2 = collect(RectilinearGrid(randn(rng, 3), randn(rng, 1)))
    atol=1e-9

    # Check that elementwise basically works.
    @test kernelmatrix_diag(k, x0, x1) isa AbstractVector
    @test length(kernelmatrix_diag(k, x0, x1)) == length(x0)

    # Check that pairwise basically works.
    @test kernelmatrix(k, x0, x2) isa AbstractMatrix
    @test size(kernelmatrix(k, x0, x2)) == (length(x0), length(x2))

    # Check that elementwise is consistent with pairwise.
    @test kernelmatrix_diag(k, x0, x1) ≈ diag(kernelmatrix(k, x0, x1)) atol=atol

    # Check additional binary elementwise properties for kernels.
    @test kernelmatrix_diag(k, x0, x1) ≈ kernelmatrix_diag(k, x1, x0)
    @test kernelmatrix(k, x0, x2) ≈ kernelmatrix(k, x2, x0)' atol=atol

    # Check that unary elementwise basically works.
    @test kernelmatrix_diag(k, x0) isa AbstractVector
    @test length(kernelmatrix_diag(k, x0)) == length(x0)

    # Check that unary pairwise basically works.
    @test kernelmatrix(k, x0) isa AbstractMatrix
    @test size(kernelmatrix(k, x0)) == (length(x0), length(x0))
    @test kernelmatrix(k, x0) ≈ kernelmatrix(k, x0)' atol=atol

    # Check that unary elementwise is consistent with unary pairwise.
    @test kernelmatrix_diag(k, x0) ≈ diag(kernelmatrix(k, x0)) atol=atol

    # Check that unary pairwise produces a positive definite matrix (approximately).
    @test all(eigvals(Matrix(kernelmatrix(k, x0))) .> -atol)

    # Check that unary elementwise / pairwise are consistent with the binary versions.
    @test kernelmatrix_diag(k, x0) ≈ kernelmatrix_diag(k, x0, x0) atol=atol
    @test kernelmatrix(k, x0) ≈ kernelmatrix(k, x0, x0) atol=atol
end
