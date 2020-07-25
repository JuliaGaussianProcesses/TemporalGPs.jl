@testset "finite_lti_sde" begin
    f = GP(Matern52(), GPC())
    x = RegularSpacing(0.0, 0.2, 15)
    σ² = randn(length(x)).^2 .+ 1e-2
    fx = f(x, σ²)

    f_sde = to_sde(f)
    fx_sde = f_sde(x, σ²)

    # Ensure that the interfaces have been implemented in the same way.
    @test length(fx) == length(fx_sde)
    @test mean(fx) ≈ mean(fx_sde)
    @test cov(fx) ≈ cov(fx_sde)

    @test rand(MersenneTwister(0), fx) ≈ rand(MersenneTwister(0), fx_sde)

    y = rand(fx_sde)
    @test logpdf(fx, y) ≈ logpdf(fx_sde, y)
end
