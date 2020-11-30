using TemporalGPs: smooth, predict, update_decorrelate, step_decorrelate, update_correlate,
    step_correlate, LGSSM, GaussMarkovModel, Gaussian

using Stheno: GP, GPC
using TemporalGPs: storage_type, is_of_storage_type, is_time_invariant
using Zygote, StaticArrays

println("lgssm:")
@testset "lgssm" begin

    @testset "mean and cov" begin
        rng = MersenneTwister(123456)
        Dlat = 3
        Dobs = 2
        N = 5
        model = random_tv_lgssm(rng, Dlat, Dobs, N, ArrayStorage(Float64))
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
        Dobss = [1, 2, 4]
        storages = [
            (name="dense storage Float64", val=ArrayStorage(Float64)),
            (name="static storage Float64", val=SArrayStorage(Float64)),
            (name="dense storage Float32", val=ArrayStorage(Float32)),
            (name="static storage Float32", val=SArrayStorage(Float32)),
        ]

        @testset "(time_varying=$tv, Dlat=$Dlat, Dobs=$Dobs, $(storage.name))" for
            tv in tvs,
            Dlat in Dlats,
            Dobs in Dobss,
            storage in storages

            # Print current iteration to prevent CI timing out.
            println("(time_varying=$tv, Dlat=$Dlat, Dobs=$Dobs, $(storage.name))")

            # Build LGSSM.
            model = tv ?
                random_tv_lgssm(rng, Dlat, Dobs, N, storage.val) :
                random_ti_lgssm(rng, Dlat, Dobs, N, storage.val)
            gmm = model.gmm
            Σs = model.Σ
            As, as, Qs, Hs, hs, x = gmm.A, gmm.a, gmm.Q, gmm.H, gmm.h, gmm.x0

            ε = convert(eltype(model), 1e-6)
            gaussian_model = Gaussian(mean(model), cov(model) + ε * I)

            # Verify that model properties are as requested.
            @test eltype(model) == eltype(storage.val)
            @test storage_type(model) == storage.val

            @test length(model) == N
            @test getindex(model, N) == (gmm = model.gmm[N], Σ = model.Σ[N])

            @test is_of_storage_type(model, storage.val)
            @test is_time_invariant(model) == 1 - tv

            # Generate a sample from the model.
            y = rand(MersenneTwister(123456), model)
            y_gauss = rand(MersenneTwister(123456), gaussian_model)
            @test vcat(y...) ≈ y_gauss atol=1e-3 rtol=1e-3
            @test y isa AbstractVector{<:AbstractVector{eltype(model)}}

            # Compute the log marginal likelihood of the observation.
            @test logpdf(model, y) ≈ logpdf(gaussian_model, vcat(y...)) atol=1e-4 rtol=1e-4
            @test logpdf(model, y) isa eltype(model)

            # Verify that whiten and unwhiten are each others inverse.
            α = TemporalGPs.whiten(model, y)
            @test TemporalGPs.unwhiten(model, α) ≈ y
            @test α isa AbstractVector{<:AbstractVector{eltype(model)}}

            lml_, y_ = TemporalGPs.logpdf_and_rand(MersenneTwister(123456), model)
            @test lml_ ≈ logpdf(model, y)
            @test y_ ≈ y
            @test lml_ isa eltype(model)
            @test y_ isa AbstractVector{<:AbstractVector{eltype(model)}}

            # Verify posterior marginal computation and sampling.
            let
                # Compute mean, covariance, and noisy-covariance naively.
                m_naive = gaussian_model.m
                Kyy = Symmetric(gaussian_model.P)
                Kff = cov(LGSSM(gmm, zero.(Σs)))

                # Compute the posterior mean and covariance naively.
                m_post_naive = m_naive + Kff * (cholesky(Kyy) \ (y_gauss - m_naive))
                Kff_post_naive = Kff - Kff * (cholesky(Kyy) \ Kff)

                # Verify posterior marginal statistics.
                _, f_smooth, _ = smooth(model, y)

                # Check posteior mean.
                m_post = vcat([f.m for f in f_smooth]...)
                @test isapprox(m_post, m_post_naive; atol=1e-5, rtol=1e-5)

                # Check posterior covariance.
                σ²_post = [f.P for f in f_smooth]
                pos = 1
                for n in 1:N
                    @test isapprox(
                        Kff_post_naive[pos:(pos + Dobs - 1), pos:(pos + Dobs - 1)],
                        σ²_post[n];
                        atol=1e-5,
                        rtol=1e-5,
                    )
                    pos += Dobs
                end

                # # Approximately verify posterior statistics via Monte Carlo.
                # # These tests are a bit crap, and I would appreciate advice regarding
                # # how to test them properly.
                # N_samples = 1_000_000
                # post_ys_raw = [posterior_rand(rng, model, y, 1) for _ in 1:N_samples]
                # post_ys = hcat([vec(vcat(post_ys_raw[s]...)) for s in 1:N_samples]...)
                # m̂_post = vec(mean(post_ys; dims=2))
                # @show size(m̂_post)
                # Kff_post_approx = (post_ys .- m_post_naive) * (post_ys .- m_post_naive)' ./ N_samples
                # @show size(Kff_post_approx)

                # display(abs.(m̂_post - m_post_naive))
                # println()
                # @test all(abs.(m̂_post - m_post_naive) .< 2e-2)
                # display(abs.(Kff_post_approx - Kff_post_naive))
                # println()
                # @test all(abs.(Kff_post_approx - Kff_post_naive) .< 2e-2)
            end

            # Compute square roots of psd matrices for finite differencing safety.
            sqrt_Qs = map(Q->cholesky(Symmetric(Q + 1e-2I)).U, Qs)
            sqrt_Σs = map(Σ->cholesky(Symmetric(Σ)).U, Σs)

            if eltype(model) == Float64

                # Verify the gradients w.r.t. sampling from the model.
                P_U = cholesky(x.P).U
                P_U = storage.val isa SArrayStorage ? UpperTriangular(SMatrix(P_U.data)) : P_U
                adjoint_test(
                    (As, as, sqrt_Qs, Hs, hs, m, sqrt_P, sqrt_Σs) -> begin
                        Qs = map(U->UpperTriangular(U)'UpperTriangular(U), sqrt_Qs)
                        Σs = map(U->UpperTriangular(U)'UpperTriangular(U), sqrt_Σs)
                        P = UpperTriangular(sqrt_P)'UpperTriangular(sqrt_P)
                        x = Gaussian(m, P)
                        gmm = GaussMarkovModel(As, as, Qs, Hs, hs, x)
                        model = LGSSM(gmm, Σs)
                        return rand(MersenneTwister(123456), model)
                    end,
                    [randn(rng, Dobs) for _ in 1:N],
                    As, as, sqrt_Qs, Hs, hs, x.m, P_U, sqrt_Σs;
                    rtol=1e-6, atol=1e-6,
                )

                # Verify the gradients w.r.t. computing the logpdf of the model.
                adjoint_test(
                    (As, as, sqrt_Qs, Hs, hs, m, sqrt_P, sqrt_Σs, y) -> begin
                        Qs = map(U->UpperTriangular(U)'UpperTriangular(U), sqrt_Qs)
                        Σs = map(U->UpperTriangular(U)'UpperTriangular(U), sqrt_Σs)
                        P = UpperTriangular(sqrt_P)'UpperTriangular(sqrt_P)
                        x = Gaussian(m, P)
                        gmm = GaussMarkovModel(As, as, Qs, Hs, hs, x)
                        model = LGSSM(gmm, Σs)
                        return logpdf(model, y)
                    end,
                    randn(rng),
                    As, as, sqrt_Qs, Hs, hs, x.m, P_U, sqrt_Σs, y;
                    atol=1e-5, rtol=1e-5,
                )

                # Verify the gradients w.r.t. whiten
                adjoint_test(
                    (As, as, sqrt_Qs, Hs, hs, m, sqrt_P, sqrt_Σs, y) -> begin
                        Qs = map(U->UpperTriangular(U)'UpperTriangular(U), sqrt_Qs)
                        Σs = map(U->UpperTriangular(U)'UpperTriangular(U), sqrt_Σs)
                        P = UpperTriangular(sqrt_P)'UpperTriangular(sqrt_P)
                        x = Gaussian(m, P)
                        gmm = GaussMarkovModel(As, as, Qs, Hs, hs, x)
                        model = LGSSM(gmm, Σs)
                        return TemporalGPs.whiten(model, y)
                    end,
                    [randn(rng, Dobs) for _ in 1:N],
                    As, as, sqrt_Qs, Hs, hs, x.m, P_U, sqrt_Σs, y;
                    atol=1e-6, rtol=1e-6,
                )

                # Verify the gradients w.r.t. unwhiten
                adjoint_test(
                    (As, as, sqrt_Qs, Hs, hs, m, sqrt_P, sqrt_Σs, α) -> begin
                        Qs = map(U->UpperTriangular(U)'UpperTriangular(U), sqrt_Qs)
                        Σs = map(U->UpperTriangular(U)'UpperTriangular(U), sqrt_Σs)
                        P = UpperTriangular(sqrt_P)'UpperTriangular(sqrt_P)
                        x = Gaussian(m, P)
                        gmm = GaussMarkovModel(As, as, Qs, Hs, hs, x)
                        model = LGSSM(gmm, Σs)
                        return TemporalGPs.unwhiten(model, α)
                    end,
                    [randn(rng, Dobs) for _ in 1:N],
                    As, as, sqrt_Qs, Hs, hs, x.m, P_U, sqrt_Σs, y;
                    atol=1e-6, rtol=1e-6,
                )
            end
        end
    end

    @testset "posterior_rand_close_data" begin
        f = to_sde(GP(Matern52(), GPC()))
        x = RegularSpacing(0.0, 1e-6, 1000)
        y = rand(f(x, 0.1))
        rng = MersenneTwister(123456)
        lgssm = TemporalGPs.build_lgssm(f(x, 0.1))
        TemporalGPs.posterior_rand(rng, lgssm, y)
    end
end
