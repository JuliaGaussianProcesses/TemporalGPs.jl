using TemporalGPs: RectilinearGrid, Separable, is_of_storage_type

@testset "to_gauss_markov" begin
    rng = MersenneTwister(123456)
    Nr = 3
    Nt = 5
    Nt_pr = 2

    @testset "restructure" begin
        adjoint_test(
            x -> TemporalGPs.restructure(x, [26, 24, 20, 30]), (randn(100), );
            check_inferred=false,
        )
        adjoint_test(
            x -> TemporalGPs.restructure(x, [26, 24, 20, 30]), (Fill(randn(), 100), );
            check_inferred=false,
        )
    end

    k_sep = 1.5 * Separable(
        SEKernel() ∘ ScaleTransform(1.4), Matern32Kernel() ∘ ScaleTransform(1.3),
    )

    σ²s = [
        (name="scalar", val=(0.1,)),
        (name="nothing", val=(1e-4, )),
    ]

    kernels = [
        (name="separable", val=k_sep),
        (name="sum-separable", val=k_sep + k_sep),
    ]

    ts = (
        (name="irregular spacing", val=sort(rand(rng, Nt))),
        (name="regular spacing", val=RegularSpacing(0.0, 0.3, Nt)),
    )

    @testset "k = $(k.name), σ²=$(σ².val), t=$(t.name)" for
        k in kernels,
        σ² in σ²s,
        t in ts

        println("k = $(k.name), σ²=$(σ².val), t=$(t.name)")

        r = randn(rng, Nr)
        x = RectilinearGrid(r, t.val)

        f = GP(k.val)
        ft = f(collect(x), σ².val...)

        f_sde = to_sde(f)
        ft_sde = f_sde(x, σ².val...)

        @test length(ft_sde) == length(x)

        y = rand(MersenneTwister(123456), ft_sde)

        model = TemporalGPs.build_lgssm(ft_sde)
        @test all(
            isequal.(
                length.(TemporalGPs.restructure(y, model.emissions)),
                dim_out.(model.emissions),
            )
        )

        @test mean.(marginals(ft)) ≈ mean.(marginals(ft_sde))
        @test std.(marginals(ft)) ≈ std.(marginals(ft_sde))
        @test logpdf(ft, y) ≈ logpdf(ft_sde, y)

        # Test that the SDE posterior is close to the naive posterior.
        f_post_naive = posterior(ft, y)
        f_post_sde = posterior(f_sde(x, σ².val...), y)

        @testset "posterior $(data.name)" for data in [
            (name="same locations", inputs=x),
            (name="different locations", inputs=RectilinearGrid(r, randn(rng, Nt_pr))),
        ]
            x_pr = data.inputs
            fx_post_naive = f_post_naive(collect(x_pr), 0.1)
            fx_post_sde = f_post_sde(x_pr, 0.1)

            @test mean.(marginals(fx_post_naive)) ≈ mean.(marginals(fx_post_sde))
            @test std.(marginals(fx_post_naive)) ≈ std.(marginals(fx_post_sde))

            y_post = rand(rng, fx_post_naive)
            @test logpdf(fx_post_naive, y_post) ≈ logpdf(fx_post_sde, y_post)

            # No statistical tests run on `rand`, which seems somewhat dangerous, but there's
            # not a lot to be done about it unfortunately.
            @testset "rand" begin
                y = rand(rng, fx_post_sde)
                @test y isa AbstractVector{<:Real}
                @test length(y) == length(x_pr)
            end

        end

        # # I'm not checking correctness here, just that it runs. No custom adjoints have been
        # # written that are involved in this that aren't tested, so there should be no need
        # # to check correctness.
        # @testset "logpdf AD" begin
        #     out, pb = Zygote._pullback(NoContext(), logpdf, ft_sde, y)
        #     pb(rand_zygote_tangent(out))
        # end
        # # adjoint_test(logpdf, (ft_sde, y); fdm=central_fdm(2, 1), check_inferred=false)

        # if t.val isa RegularSpacing
        #     adjoint_test(
        #         (r, Δt, y) -> begin
        #             x = RectilinearGrid(r, RegularSpacing(t.val.t0, Δt, Nt))
        #             _f = to_sde(GP(k.val, GPC()))
        #             _ft = _f(x, σ².val...)
        #             return logpdf(_ft, y)
        #         end,
        #         (r, t.val.Δt, y_sde);
        #         check_inferred=false,
        #     )
        # end
    end
end
