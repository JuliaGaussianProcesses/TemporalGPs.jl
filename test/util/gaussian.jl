@testset "Gaussian" begin
    N = 11
    @test TemporalGPs.dim(Gaussian(randn(N), randn(N, N))) == N

    @testset "Reversion test for #56" begin
        x = Gaussian(SVector{3}(randn(3)), zeros(3, 3))
        @test rand(x) ≈ mean(x) rtol=1e-4
    end
end
