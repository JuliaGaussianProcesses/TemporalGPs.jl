using TemporalGPs:
    dtc,
    dtcify,
    DTCSeparable,
    RectilinearGrid,
    RegularInTime,
    get_times,
    get_space,
    Separable,
    approx_posterior_marginals

@testset "pseudo_point" begin

    rng = MersenneTwister(123456)

    @testset "dtcify" begin
        z = randn(rng, 3)
        k_sep = Separable(SEKernel(), Matern32Kernel())
        @test dtcify(z, k_sep) isa DTCSeparable
        @test dtcify(z, 0.5 * k_sep) isa ScaledKernel{<:DTCSeparable}
        @test dtcify(z, transform(k_sep, 0.5)) isa TransformedKernel{<:DTCSeparable}
        @test dtcify(z, k_sep + k_sep) isa KernelSum{<:Tuple{DTCSeparable, DTCSeparable}}
    end

    # A couple of "base" kernels used as components in more complicated kernels below.
    separable_1 = Separable(SEKernel(), Matern12Kernel())
    separable_2 = Separable(SEKernel(), Matern52Kernel())

    # The various spatio-temporal kernels to try out.
    kernels = [

        (name="separable-1", val=separable_1),
        (name="separable-2", val=separable_2),

        (name="scaled-separable", val=0.5 * Separable(Matern52Kernel(), Matern32Kernel())),
        (
            name="stretched-separable",
            val=Separable(SEKernel(), transform(Matern12Kernel(), 1.3)),
        ),

        (name="sum-separable-1", val=separable_1 + separable_2),
        (name="sum-separable-2", val=1.3 * separable_1 + 0.95 * separable_2),
    ]

    # Input locations.
    xs = [
        (
            name="rectilinear",
            val=RectilinearGrid(randn(rng, 2), RegularSpacing(0.0, 0.3, 3)),
        ),
        (
            name="regular-in-time",
            val=RegularInTime(
                RegularSpacing(0.0, 0.1, 11),
                [randn(rng, 3) for _ in 1:11],
            ),
        ),
    ]

    # Spatial-locations of pseudo-inputs and predictions.

    z_r = randn(rng, 2)
    x_pr_r = randn(rng, 10)

    @testset "kernel=$(k.name), x=$(x.name)" for k in kernels, x in xs



        # Compute pseudo-input locations. These have to share time points with `x`.
        t = get_times(x.val)
        z = RectilinearGrid(z_r, t)
        z_naive = collect(z)

        # Construct naive GP.
        f_naive = GP(k.val)
        fx_naive = f_naive(collect(x.val), 0.1)
        y = rand(rng, fx_naive)

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
        @test dtc_naive ≈ dtc_sde rtol=1e-6

        elbo_naive = elbo(fx_naive, y, f_naive(z_naive))
        elbo_sde = elbo(fx, y, z_r)
        @test elbo_naive ≈ elbo_sde rtol=1e-6

        adjoint_test(
            (y, z_r) -> elbo(fx, y, z_r), (y, z_r);
            rtol=1e-7,
            context=Zygote.Context(),
            check_infers=false,
        )

        # Compute approximate posterior marginals naively.
        f_approx_post_naive = approx_posterior(VFE(), fx_naive, y, f_naive(z_naive))
        x_pr = RectilinearGrid(x_pr_r, get_times(x.val))
        naive_approx_post_marginals = marginals(f_approx_post_naive(collect(x_pr)))

        # This is a horrible interface, but it's the best that I can do on short notice.
        approx_post_marginals = approx_posterior_marginals(dtc, fx, y, z_r, x_pr_r)

        @test mean.(naive_approx_post_marginals) ≈ mean.(approx_post_marginals) rtol=1e-7
        @test std.(naive_approx_post_marginals) ≈ std.(approx_post_marginals) rtol=1e-7

        # Similarly awful interface, make predictions for each point separately.
        approx_post_marginals_individual = map(eachindex(t)) do t
            approx_posterior_marginals(dtc, fx, y, z_r, x_pr_r, t)
        end

        approx_post_marginals_vec = reduce(vcat, approx_post_marginals_individual)

        @test mean.(approx_post_marginals) ≈ mean.(approx_post_marginals_vec) rtol=1e-7
        @test std.(approx_post_marginals) ≈ std.(approx_post_marginals_vec) rtol=1e-7

        # Do the RegularInTime one and compare it against RectilinearGrid.
        x_pr_rit = RegularInTime(get_times(x_pr), [get_space(x_pr) for _ in get_times(x.val)])
        approx_post_marginals_rit = approx_posterior_marginals(dtc, fx, y, z_r, x_pr_rit)

        @test mean.(approx_post_marginals) ≈ mean.(approx_post_marginals_rit) rtol=1e-7
        @test std.(approx_post_marginals) ≈ std.(approx_post_marginals_rit) rtol=1e-7

        @testset "missings" begin

            # Construct missing data.
            y_missing = Vector{Union{eltype(y), Missing}}(undef, size(y))
            y_missing .= y
            missing_idx = sort(randperm(rng, length(y))[1:Int(floor(length(y) / 3))])
            missing_idx = eachindex(y)
            y_missing[missing_idx] .= missing

            # Construct naive missing inputs and outputs.
            present_idx = setdiff(eachindex(y), missing_idx)
            naive_inputs_missings = collect(x.val)[present_idx]
            naive_y_missings = y[present_idx]
            fx_naive = f_naive(naive_inputs_missings, 0.1)

            # Compute DTC using both approaches.
            dtc_naive = dtc(fx_naive, naive_y_missings, f_naive(z_naive))
            dtc_sde = dtc(fx, y_missing, z_r)
            @test dtc_naive ≈ dtc_sde rtol=1e-7 atol=1e-7

            elbo_naive = elbo(fx_naive, naive_y_missings, f_naive(z_naive))
            elbo_sde = elbo(fx, y_missing, z_r)
            @test elbo_naive ≈ elbo_sde rtol=1e-7 atol=1e-7

            # Compute approximate posterior marginals naively with missings.
            f_approx_post_naive = approx_posterior(
                VFE(), fx_naive, naive_y_missings, f_naive(z_naive),
            )
            naive_approx_post_marginals = marginals(f_approx_post_naive(collect(x_pr)))

            # Compute approximate posterior marginals using the state-space approximation.
            approx_post_marginals = approx_posterior_marginals(
                dtc, fx, y_missing, z_r, x_pr_r,
            )

            @test mean.(naive_approx_post_marginals) ≈ mean.(approx_post_marginals) rtol=1e-7
            @test std.(naive_approx_post_marginals) ≈ std.(approx_post_marginals) rtol=1e-7
        end
    end
end
