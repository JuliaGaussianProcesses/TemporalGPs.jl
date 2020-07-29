@testset "posterior_lti_sde" begin
    f = GP(Matern52(), GPC())
    x = RegularSpacing(0.0, 0.2, 15)
    σ² = randn(length(x)).^2 .+ 1e-2
    x_pred = collect(range(-3.0, 18.0; length=100))
    fx = f(x, σ²)
    y = rand(fx)

    f_post = f | (fx ← y)
    fx_post = f_post(x_pred, 0.1)
 
    f_sde_post = posterior(to_sde(f)(x, σ²), y)
    fx_sde_post = f_sde_post(x_pred, 0.1)


    # Ensure that the interfaces have been implemented in the same way.
    @test length(fx_post) == length(fx_sde_post)
    @test mean(fx_post) ≈ mean(fx_sde_post)
    @test mean.(marginals(fx_post)) ≈ mean.(marginals(fx_sde_post))
    @test std.(marginals(fx_post)) ≈ std.(marginals(fx_sde_post))

    # @test rand(MersenneTwister(0), fx_post) ≈ rand(MersenneTwister(0), fx_sde_post)

    y_pr = rand(fx_sde_post)
    @test logpdf(fx_post, y_pr) ≈ logpdf(fx_sde_post, y_pr)
end
