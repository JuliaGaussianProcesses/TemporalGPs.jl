# @testset "posterior_lti_sde" begin
#     f = GP(Matern52(), GPC())
#     x = RegularSpacing(0.0, 0.2, 15)
#     σ² = randn(length(x)).^2 .+ 1e-2
#     x_pred = collect(range(-3.0, 18.0; length=100))
#     fx = f(x, σ²)
#     y = rand(fx)

#     f_post = f | (fx ← y)
#     fx_post = f_post(x_pred, 0.1)
 
#     f_sde_post = posterior(to_sde(f)(x, σ²), y)
#     fx_sde_post = f_sde_post(x_pred, 0.1)


#     # Ensure that the interfaces have been implemented in the same way.
#     @test length(fx_post) == length(fx_sde_post)
#     @test mean(fx_post) ≈ mean(fx_sde_post)
#     @test mean.(marginals(fx_post)) ≈ mean.(marginals(fx_sde_post))
#     @test std.(marginals(fx_post)) ≈ std.(marginals(fx_sde_post))

#     # @test rand(MersenneTwister(0), fx_post) ≈ rand(MersenneTwister(0), fx_sde_post)

#     y_pr = rand(fx_sde_post)
#     @test logpdf(fx_post, y_pr) ≈ logpdf(fx_sde_post, y_pr)
# end

@testset "posterior_lti_sde" begin
    rng = MersenneTwister(123456)
    N = 13
    Npr = 15

    kernels = vcat(

        # Base kernels.
        (name="base-Matern12", val=Matern12()),
        map([Matern32, Matern52]) do k
            (name="base-$k", val=k())
        end,

        # Scaled kernels.
        map([1e-1, 1.0, 10.0, 100.0]) do σ²
            (name="scaled-σ²=$σ²", val=σ² * Matern32())
        end,

        # Stretched kernels.
        map([1e-2, 0.1, 1.0, 10.0, 100.0]) do λ
            (name="stretched-λ=$λ", val=stretch(Matern32(), λ))
        end,

        # Summed kernels.
        (
            name="sum-Matern12-Matern32",
            val=1.5 * stretch(Matern12(), 0.1) + 0.3 * stretch(Matern32(), 1.1),
        ),
    )

    # Construct a Gauss-Markov model with either dense storage or static storage.
    storages = (
        (name="dense storage Float64", val=ArrayStorage(Float64)),
        (name="static storage Float64", val=SArrayStorage(Float64)),
        # (name="dense storage Float32", val=ArrayStorage(Float32)),
        # (name="static storage Float32", val=SArrayStorage(Float32)),
    )

    # Either regular spacing or irregular spacing in time.
    ts = (
        (name="irregular spacing", val=collect(RegularSpacing(0.0, 0.3, N))),
        (name="regular spacing", val=RegularSpacing(0.0, 0.3, N)),
    )

    σ²s = (
        (name="homoscedastic noise", val=(0.1, ),),
        (name="heteroscedastic noise", val=(rand(rng, N) .+ 1e-1, )),
        (name="none", val=()),
    )

    @testset "$(kernel.name), $(storage.name), $(t.name), $(σ².name)" for
        kernel in kernels,
        storage in storages,
        t in ts,
        σ² in σ²s

        println("$(kernel.name), $(storage.name), $(t.name), $(σ².name)")

        # Construct Gauss-Markov model.
        f_naive = GP(kernel.val, GPC())
        fx_naive = f_naive(collect(t.val), σ².val...)

        f = to_sde(f_naive, storage.val)
        fx = f(t.val, σ².val...)
        model = build_lgssm(fx)

        # is_of_storage_type(fx, storage.val)
        validate_dims(model)

        y = rand(rng, fx)

        x_pr = rand(rng, Npr) * (maximum(t.val) - minimum(t.val)) .+ minimum(t.val)

        f_post_naive = f_naive | (fx_naive ← y)
        f_post = posterior(fx, y)

        post_obs_var = 0.1
        fx_post_naive = f_post_naive(x_pr, post_obs_var)
        fx_post = f_post(x_pr, post_obs_var)
        y_post = rand(fx_post)

        @test mean.(marginals(fx_post_naive)) ≈ mean.(marginals(fx_post)) rtol=1e-5
        @test std.(marginals(fx_post_naive)) ≈ std.(marginals(fx_post)) rtol=1e-5
        @test logpdf(fx_post_naive, y_post) ≈ logpdf(fx_post, y_post) rtol=1e-5
    end
end
