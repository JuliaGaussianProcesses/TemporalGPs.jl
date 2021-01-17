using TemporalGPs:
    dtc,
    dtcify,
    DTCSeparable,
    RectilinearGrid,
    RegularInTime,
    get_time,
    get_space,
    Separable,
    approx_posterior_marginals

@testset "pseudo_point" begin
    @testset "dtcify" begin
        z = randn(3)
        k_sep = Separable(EQ(), Matern32())
        @test dtcify(z, k_sep) isa DTCSeparable
        @test dtcify(z, 0.5 * k_sep) isa Stheno.Scaled{<:Any, <:DTCSeparable}
        @test dtcify(z, stretch(k_sep, 0.5)) isa Stheno.Stretched{<:Any, <:DTCSeparable}
        @test dtcify(z, k_sep + k_sep) isa Stheno.Sum{<:DTCSeparable, <:DTCSeparable}
    end

    # A couple of "base" kernels used as components in more complicated kernels below.
    separable_1 = Separable(EQ(), Matern12())
    separable_2 = Separable(EQ(), Matern52())

    # The various spatio-temporal kernels to try out.
    kernels = [

        (name="separable-1", val=separable_1),
        (name="separable-2", val=separable_2),

        (name="scaled-separable", val=0.5 * Separable(Matern52(), Matern32())),
        (name="stretched-separable", val=Separable(EQ(), stretch(Matern12(), 1.3))),

        (name="sum-separable-1", val=separable_1 + separable_2),
        (name="sum-separable-2", val=1.3 * separable_1 + separable_2 * 0.95),
    ]

    # Input locations.
    xs = [
        (
            name="rectilinear",
            val=RectilinearGrid(randn(25), RegularSpacing(0.0, 0.3, 10)),
        ),
        (
            name="regular-in-time",
            val=RegularInTime(
                RegularSpacing(0.0, 0.1, 11),
                [randn(3) for _ in 1:11],
            ),
        ),
    ]

    # Spatial-locations of pseudo-inputs.
    z_r = randn(2)
    x_pr_r = randn(10)

    @testset "kernel=$(k.name), x=$(x.name)" for k in kernels, x in xs

        # Compute pseudo-input locations. These have to share time points with `x`.
        t = get_time(x.val)
        z = RectilinearGrid(z_r, t)
        z_naive = collect(z)

        # Construct naive GP.
        f_naive = GP(k.val, GPC())
        fx_naive = f_naive(collect(x.val), 0.1)
        y = rand(fx_naive)

        # Construct state-space GP.
        f = to_sde(f_naive)
        fx = f(x.val, 0.1)

        # Verify dimensions of the LGSSM constructed to compute the DTC. This catches a
        # surprisingly large number of bugs during development.
        fx_dtc = TemporalGPs.dtcify(z_r, fx)
        lgssm = TemporalGPs.build_lgssm(fx_dtc)
        @test sum(map(dim_out, lgssm.emissions)) == length(y)
        validate_dims(lgssm)

        # The two approaches to DTC computation should be equivalent up to roundoff error.
        dtc_naive = dtc(fx_naive, y, f_naive(z_naive))
        dtc_sde = dtc(fx, y, z_r)
        @test dtc_naive ≈ dtc_sde rtol=1e-7

        elbo_naive = elbo(fx_naive, y, f_naive(z_naive))
        elbo_sde = elbo(fx, y, z_r)
        @test elbo_naive ≈ elbo_sde rtol=1e-7

        # Compute approximate posterior marginals naively.
        f_approx_post_naive = f_naive | Stheno.PseudoObs(fx_naive ← y, f_naive(z_naive))
        x_pr = RectilinearGrid(x_pr_r, get_time(x.val))
        naive_approx_post_marginals = marginals(f_approx_post_naive(collect(x_pr)))

        # This is a horrible interface, but it's the best that I can do on short notice.
        approx_post_marginals = approx_posterior_marginals(dtc, fx, y, z_r, x_pr_r)

        @test mean.(naive_approx_post_marginals) ≈ mean.(approx_post_marginals) rtol=1e-7
        @test std.(naive_approx_post_marginals) ≈ std.(approx_post_marginals) rtol=1e-7
    end
end
