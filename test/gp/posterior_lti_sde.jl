@testset "posterior_lti_sde" begin
    rng = MersenneTwister(123456)
    N = 13
    Npr = 15

    kernels = vcat(

        # Base kernels.
        (name="base-Matern12", val=Matern12Kernel()),
        map([Matern32Kernel, Matern52Kernel]) do k
            (name="base-$k", val=k())
        end,

        # Scaled kernels.
        map([1e-1, 1.0, 10.0, 100.0]) do σ²
            (name="scaled-σ²=$σ²", val=σ² * Matern32Kernel())
        end,

        # Stretched kernels.
        map([1e-2, 0.1, 1.0, 10.0, 100.0]) do λ
            (name="stretched-λ=$λ", val=stretch(Matern32Kernel(), λ))
        end,

        # Summed kernels.
        (
            name="sum-Matern12-Matern32",
            val=1.5 * stretch(Matern12Kernel(), 0.1) + 0.3 * stretch(Matern32Kernel(), 1.1),
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
    )

    @testset "$(kernel.name), $(storage.name), $(t.name), $(σ².name)" for
        kernel in kernels,
        storage in storages,
        t in ts,
        σ² in σ²s

        println("$(kernel.name), $(storage.name), $(t.name), $(σ².name)")

        # Construct Gauss-Markov model.
        f_naive = GP(kernel.val)
        fx_naive = f_naive(collect(t.val), σ².val...)

        f = to_sde(f_naive, storage.val)
        fx = f(t.val, σ².val...)
        model = build_lgssm(fx)

        validate_dims(model)

        y = rand(rng, fx)

        x_pr = rand(rng, Npr) * (maximum(t.val) - minimum(t.val)) .+ minimum(t.val)

        f_post_naive = posterior(fx_naive, y)
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
