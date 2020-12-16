using TemporalGPs: RectilinearGrid, Separable, is_of_storage_type, is_time_invariant

@testset "to_gauss_markov" begin
    rng = MersenneTwister(123456)
    Nr = 3
    Nt = 5

    k_sep = 1.5 * Separable(stretch(EQ(), 1.4), stretch(Matern32(), 1.3))

    σ²s = [
        (name="scalar", val=(0.1,)),
        (name="nothing", val=()),
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

        r = randn(rng, Nr)
        x = RectilinearGrid(r, t.val)

        f = GP(k.val, GPC())
        ft = f(collect(x), σ².val...)
        y = rand(rng, ft)

        f_sde = to_sde(f)
        ft_sde = f_sde(x, σ².val...)

        should_be_time_invariant = (t.val isa Vector) ? false : true
        @test is_time_invariant(ft_sde) == should_be_time_invariant
        @test is_of_storage_type(ft_sde, ArrayStorage(Float64))

        validate_dims(ft_sde)
        @test length(ft_sde) == length(x.xr)

        y_naive = rand(MersenneTwister(123456), ft)
        y_sde = rand(MersenneTwister(123456), ft_sde)
        @test y_naive ≈ vcat(y_sde...)

        @test logpdf(ft, y_naive) ≈ logpdf(ft_sde, y_sde)

        if t.val isa RegularSpacing
            adjoint_test(
                (r, Δt, y) -> begin
                    x = RectilinearGrid(r, RegularSpacing(t.val.t0, Δt, Nt))
                    _f = to_sde(GP(k.val, GPC()))
                    _ft = _f(x, σ².val...)
                    return logpdf(_ft, y)
                end,
                (r, t.val.Δt, y_sde);
                check_infers=false,
            )
        end
    end
end
