using TemporalGPs: smooth, StaticStorage, DenseStorage, _predict,
    update_decorrelate, step_decorrelate, update_correlate, step_correlate, LGSSM,
    GaussMarkovModel, Gaussian
# using TemporalGPs: update_correlate_pullback, update_decorrelate_pullback,
#     step_decorrelate_pullback
using Stheno: GP, GPC
using Zygote, StaticArrays

@testset "lgssm" begin

    @testset "mean and cov" begin
        rng = MersenneTwister(123456)
        Dlat = 3
        Dobs = 2
        N = 5
        model = random_tv_lgssm(rng, Dlat, Dobs, N, DenseStorage())
        @test mean(model) == mean(model.gmm)

        P = cov(model)
        @test size(P) == (N * Dobs, N * Dobs)
        @test all(eigvals(P) .> 0)
    end

    @testset "correctness" begin
        rng = MersenneTwister(123456)
        N = 3

        tvs = [true, false]
        Dlats = [1, 3, 4]
        Douts = [1, 2, 5]
        storages = [
            (name="dense storage", val=DenseStorage()),
            (name="static storage", val=StaticStorage()),
        ]

        @testset "(time_varying=$tv, Dlat=$Dlat, Dobs=$Dout, $(storage.name))" for
            tv in tvs,
            Dlat in Dlats,
            Dout in Douts,
            storage in storages

            # Build LGSSM.
            model = tv ?
                random_tv_lgssm(rng, Dlat, Dout, N, storage.val) :
                random_ti_lgssm(rng, Dlat, Dout, N, storage.val)
            gmm = model.gmm
            Σs = model.Σ
            As, as, Qs, Hs, hs, x = gmm.A, gmm.a, gmm.Q, gmm.H, gmm.h, gmm.x0

            gaussian_model = Gaussian(mean(model), cov(model) + 1e-6I)

            # Generate a sample from the model.
            y = rand(MersenneTwister(123456), model)
            y_gauss = rand(MersenneTwister(123456), gaussian_model)
            @test vcat(y...) ≈ y_gauss atol=1e-3 rtol=1e-3

            # Compute the log marginal likelihood of the observation.
            @test logpdf(model, y) ≈ logpdf(gaussian_model, vcat(y...)) atol=1e-4 rtol=1e-4

            # Verify that whiten and unwhiten are each others inverse.
            α = TemporalGPs.whiten(model, y)
            @test TemporalGPs.unwhiten(model, α) ≈ y

            lml_, y_ = TemporalGPs.logpdf_and_rand(MersenneTwister(123456), model)
            @test lml_ ≈ logpdf(model, y)
            @test y_ ≈ y

            # Compute square roots of psd matrices for finite differencing safety.
            sqrt_Qs = map(Q->cholesky(Symmetric(Q + 1e-2I)).U, Qs)
            sqrt_Σs = map(Σ->cholesky(Symmetric(Σ)).U, Σs)

            # # Verify the gradients w.r.t. sampling from the model.
            # adjoint_test(
            #     (As, as, sqrt_Qs, Hs, hs, m, sqrt_P, sqrt_Σs) -> begin
            #         Qs = map(U->UpperTriangular(U)'UpperTriangular(U), sqrt_Qs)
            #         Σs = map(U->UpperTriangular(U)'UpperTriangular(U), sqrt_Σs)
            #         P = UpperTriangular(sqrt_P)'UpperTriangular(sqrt_P)
            #         x = Gaussian(m, P)
            #         gmm = GaussMarkovModel(As, as, Qs, Hs, hs, x)
            #         model = LGSSM(gmm, Σs)
            #         return rand(MersenneTwister(123456), model)
            #     end,
            #     [randn(rng, Dout) for _ in 1:N],
            #     As, as, sqrt_Qs, Hs, hs, x.m, cholesky(x.P).U, sqrt_Σs;
            #     rtol=1e-6, atol=1e-6,
            # )

            # # Verify the gradients w.r.t. computing the logpdf of the model.
            # adjoint_test(
            #     (As, as, sqrt_Qs, Hs, hs, m, sqrt_P, sqrt_Σs, y) -> begin
            #         Qs = map(U->UpperTriangular(U)'UpperTriangular(U), sqrt_Qs)
            #         Σs = map(U->UpperTriangular(U)'UpperTriangular(U), sqrt_Σs)
            #         P = UpperTriangular(sqrt_P)'UpperTriangular(sqrt_P)
            #         x = Gaussian(m, P)
            #         gmm = GaussMarkovModel(As, as, Qs, Hs, hs, x)
            #         model = LGSSM(gmm, Σs)
            #         return logpdf(model, y)
            #     end,
            #     randn(rng),
            #     As, as, sqrt_Qs, Hs, hs, x.m, cholesky(x.P).U, sqrt_Σs, y;
            #     atol=1e-6, rtol=1e-6,
            # )

            # # Verify the gradients w.r.t. whiten
            # adjoint_test(
            #     (As, as, sqrt_Qs, Hs, hs, m, sqrt_P, sqrt_Σs, y) -> begin
            #         Qs = map(U->UpperTriangular(U)'UpperTriangular(U), sqrt_Qs)
            #         Σs = map(U->UpperTriangular(U)'UpperTriangular(U), sqrt_Σs)
            #         P = UpperTriangular(sqrt_P)'UpperTriangular(sqrt_P)
            #         x = Gaussian(m, P)
            #         gmm = GaussMarkovModel(As, as, Qs, Hs, hs, x)
            #         model = LGSSM(gmm, Σs)
            #         return TemporalGPs.whiten(model, y)
            #     end,
            #     [randn(rng, Dout) for _ in 1:N],
            #     As, as, sqrt_Qs, Hs, hs, x.m, cholesky(x.P).U, sqrt_Σs, y;
            #     atol=1e-6, rtol=1e-6,
            # )

            # # Verify the gradients w.r.t. unwhiten
            # adjoint_test(
            #     (As, as, sqrt_Qs, Hs, hs, m, sqrt_P, sqrt_Σs, α) -> begin
            #         Qs = map(U->UpperTriangular(U)'UpperTriangular(U), sqrt_Qs)
            #         Σs = map(U->UpperTriangular(U)'UpperTriangular(U), sqrt_Σs)
            #         P = UpperTriangular(sqrt_P)'UpperTriangular(sqrt_P)
            #         x = Gaussian(m, P)
            #         gmm = GaussMarkovModel(As, as, Qs, Hs, hs, x)
            #         model = LGSSM(gmm, Σs)
            #         return TemporalGPs.unwhiten(model, α)
            #     end,
            #     [randn(rng, Dout) for _ in 1:N],
            #     As, as, sqrt_Qs, Hs, hs, x.m, cholesky(x.P).U, sqrt_Σs, y;
            #     atol=1e-6, rtol=1e-6,
            # )
        end
    end

    # @testset "static filtering gradients" begin
    #     rng = MersenneTwister(123456)
    #     t = range(0.1; step=0.11, length=10_000)
    #     f = GP(Matern32(), GPC())
    #     σ²_n = 0.54
    #     model = ssm(f(t, 0.54), StaticStorage()).model
    #     x = model.x₀
    #     D = length(x.m)
    #     m, P = x.m, x.P
    #     A, Q = first(model.A), first(model.Q)
    #     h, σ² = first(model.H), first(model.Σ)
    #     b = first(model.b)
    #     c = first(model.c)
    #     @testset "logpdf performance" begin
    #         Δlml = randn(rng)
    #         t = range(0.1; step=0.11, length=1_000_000)
    #         model = ssm(f(t, 0.54), StaticStorage())
    #         y = collect(rand(rng, model))

    #         # Ensure that allocs is roughly independent of length(t).
    #         primal, fwd, rvs = benchmark_adjoint(logpdf, Δlml, model, y; disp=false)
    #         @test allocs(primal) < 100
    #         @test allocs(fwd) < 100
    #         @test allocs(rvs) < 100
    #     end
    #     @testset "rand" begin
    #         t = range(0.1; step=0.11, length=1_000_000)
    #         Δy = randn(rng, length(t))
    #         model = ssm(f(t, 0.54), StaticStorage())

    #         # Ensure that allocs is roughly independent of length(t).
    #         primal, fwd, rvs = benchmark_adjoint(
    #             model->rand(MersenneTwister(123456), model), Δy, model;
    #             disp=false,
    #         )
    #         @test allocs(primal) < 100
    #         @test allocs(fwd) < 100
    #         @test allocs(rvs) < 100
    #     end
    # end
end
