using TemporalGPs: RectilinearGrid, Separable

@testset "to_gauss_markov" begin
    rng = MersenneTwister(123456)
    Nr = 3
    Nt = 5
    x = RectilinearGrid(randn(rng, Nr), sort(randn(rng, Nt)))
    k = Separable(EQ(), Matern32())
    σ² = 0.1

    k_sep = Separable(EQ(), Matern32())

    σ²s = [
        (name="scalar", val=(0.1,)),
        (name="nothing", val=()),
    ]

    kernels = [
        (name="separable", val=k_sep),
        (name="sum-separable", val=k_sep + k_sep),
        (
            name="sum-separable-stretched",
            val=k_sep + Separable(EQ(), stretch(Matern32(), 0.9)),
        ),
        (
            name="separable-scaled",
            val=0.5 * k_sep,
        ),
        (
            name="mixed-separable",
            val=0.5 * k_sep + 0.3 * Separable(EQ(), stretch(Matern32(), 0.95)),
        ),
    ]

    @testset "k = $(k.name), σ²=$(σ².val)" for k in kernels, σ² in σ²s

        f = GP(k.val, GPC())
        ft = f(collect(x), σ².val...)
        y = rand(rng, ft)

        f_sde = to_sde(f)
        ft_sde = f_sde(x, σ².val...)

        validate_dims(ft_sde)
        @test length(ft_sde) == length(x.xr)

        y_naive = rand(MersenneTwister(123456), ft)
        y_sde = rand(MersenneTwister(123456), ft_sde)
        @test y_naive ≈ vcat(y_sde...)

        @test logpdf(ft, y_naive) ≈ logpdf(ft_sde, y_sde)
    end
end
