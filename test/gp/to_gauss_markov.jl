using TemporalGPs: GaussMarkovModel, StaticStorage, DenseStorage

@testset "to_sde" begin

    @testset "blk_diag" begin
        adjoint_test(TemporalGPs.blk_diag, randn(5, 5), randn(2, 2), randn(3, 3))
    end

    @testset "GaussMarkovModel from kernel correctness" begin
        rng = MersenneTwister(123456)
        N = 5

        kernels_info = vcat(

            # (name="base-Matern32", ctor=()->Matern32(), θ=()),

            # Base kernels.
            map([Matern12, Matern32, Matern52]) do kernel
                (name="base-$kernel", ctor=()->kernel(), θ=())
            end,

            # Scaled kernels.
            map([1e-1, 1.0, 10.0, 100.0]) do σ²
                (name="scaled-σ²=$σ²", ctor=(σ->σ^2 * Matern32()), θ=(sqrt(σ²),))
            end,

            # Stretched kernels.
            map([1e-4, 0.1, 1.0, 10.0, 100.0]) do λ
                (name="stretched-λ=$λ", ctor=(λ->stretch(Matern32(), λ)), θ=(λ,))
            end,

            # Summed kernels.
            (
                name="sum-Matern12-Matern32",
                ctor=(λl, λr, σl, σr)->begin
                    k_l = σl^2 * stretch(Matern12(), λl)
                    k_r = σr^2 * stretch(Matern32(), λr)
                    return k_l + k_r
                end,
                θ=(0.1, 1.1, 1.5, 0.3),
            ),
        )

        # construct a Gauss-Markov model with either dense storage or static storage.
        storages = (
            (name="dense storage", val=DenseStorage()),
            (name="static storage", val=StaticStorage()),
        )

        # Either regular spacing or irregular spacing in time.
        ts = (
            (name="irregular spacing", val=sort(rand(rng, N))),
            # (name="regular spacing", val=range(0.0; step=0.3, length=N)),
        )

        @testset "$(kernel_info.name), $(storage.name), $(t.name)" for
                kernel_info in kernels_info,
                storage in storages,
                t in ts

            # Construct Gauss-Markov model.
            k = kernel_info.ctor(kernel_info.θ...)
            ft = GaussMarkovModel(k, t.val, storage.val)

            # Check that the covariances agree.
            @test cov(ft) ≈ pw(k, t.val, t.val)

            # Ensure that it's possible to backprop through construction.
            if length(kernel_info.θ) > 0
                N = length(ft)
                Dobs = size(first(ft.H), 1)
                Dlat = size(first(ft.H), 2)
                ΔA = map(_ -> randn(rng, Dlat, Dlat), 1:N)
                ΔQ = map(_ -> random_nice_psd_matrix(rng, Dlat, storage.val), 1:N)
                ΔH = map(_ -> randn(rng, Dobs, Dlat), 1:N)
                ΔH_sum = randn(rng, Dobs, Dlat)
                Δm = randn(rng, size(ft.x0.m))
                ΔP = random_nice_psd_matrix(rng, Dlat, storage.val)

                adjoint_test(
                    (θ) -> begin
                        k = kernel_info.ctor(θ...)
                        ft = GaussMarkovModel(k, t.val, storage.val)
                        return (ft.A, ft.Q, sum(ft.H), ft.x0.m, ft.x0.P)
                    end,
                    (ΔA, ΔQ, ΔH_sum, Δm, ΔP),
                    kernel_info.θ;
                )
            end
        end
    end

    @testset "static perf" begin
        k = Matern32()
        t = range(0.0; step=0.3, length=11)
        @test (@ballocated TemporalGPs.GaussMarkovModel($k, $t, StaticStorage())) == 0
    end
end









    # @testset "(de)correlate" begin
    #     rng = MersenneTwister(123456)

    #     x = range(0.0; step=0.11, length=50)
    #     f = GP(Matern52(), GPC())
    #     fx = f(x, 0.1)

    #     f_sde = to_sde(f)
    #     fx_sde = f_sde(x, SMatrix{1, 1}(0.1))

    #     U = cholesky(cov(fx)).U
    #     y = rand(MersenneTwister(123456), fx)

    #     α = TemporalGPs.whiten(fx_sde, y)
    #     y′ = TemporalGPs.unwhiten(fx_sde, α)

    #     @test y ≈ y′

    #     v = randn(rng, length(x))
    #     @test TemporalGPs.whiten(fx_sde, v) ≈ U' \ v
    #     @test TemporalGPs.unwhiten(fx_sde, v) ≈ U' * v
    # end

    # @testset "rand (statistical)" begin

    #     # Specify a GP through time.
    #     rng = MersenneTwister(123456)
    #     x = range(0.0; step=0.11, length=5)
    #     f = GP(Matern32(), GPC())
    #     fx = f(x, 0.54)

    #     # Sample from LGSSM equivalent of GP prior at x lots of times.
    #     fx_sde = to_sde(f, StaticStorage())(x, SMatrix{1, 1}(0.54))
    #     ys = [first.(rand(rng, fx_sde)) for _ in 1:1_000_000]

    #     # Check the mean and covariance roughly agree.
    #     @test all(isapprox.(mean(ys), 0.0; atol=1e-2, rtol=1e-2))
    #     @test all(isapprox.(cov(ys), cov(fx); atol=1e-2, rtol=1e-2))

    #     # Check that DenseStorage produces the same numbers.
    #     y_dense = rand(MersenneTwister(123456), to_sde(f)(x, SMatrix{1, 1}(0.54)))
    #     y_static = rand(MersenneTwister(123456), fx_sde)
    #     @test y_dense ≈ y_static
    # end
