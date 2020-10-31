using TemporalGPs: smooth, Gaussian, GaussMarkovModel, is_of_storage_type, is_time_invariant

println("scalar_lgssm:")
@testset "scalar_lgssm" begin

    @testset "from_vector_observations pullback" begin
        N = 10
        Δy_reals = randn(N)

        @testset "Vector" begin
            y_vecs = [randn(1) for _ in 1:N]
            adjoint_test(
                TemporalGPs.from_vector_observations,
                Δy_reals,
                y_vecs,
            )
        end

        @testset "SVector" begin
            y_vecs = [SVector{1}(randn()) for _ in 1:N]
            adjoint_test(
                TemporalGPs.from_vector_observations,
                Δy_reals,
                y_vecs,
            )
        end
    end

    rng = MersenneTwister(123456)
    N = 3

    Dlats = [1, 3, 4]
    storages = [
        (name="dense storage", val=ArrayStorage(Float64)),
        (name="static storage", val=SArrayStorage(Float64)),
    ]

    @testset "(Dlat=$Dlat, $(storage.name))" for
        Dlat in Dlats,
        storage in storages

        # Build LGSSM.
        scalar_model = random_tv_scalar_lgssm(rng, Dlat, N, storage.val)
        model = scalar_model.model
        gmm = model.gmm
        Σs = model.Σ
        As, as, Qs, Hs, hs, x = gmm.A, gmm.a, gmm.Q, gmm.H, gmm.h, gmm.x0

        @test is_of_storage_type(scalar_model, storage.val)
        @test is_time_invariant(scalar_model) == false

        # Generate a sample from the model.
        y = rand(MersenneTwister(123456), scalar_model)
        y_vec = rand(MersenneTwister(123456), model)
        @test y == first.(y_vec)

        # Compute the log marginal likelihood of the observation.
        @test logpdf(scalar_model, y) == logpdf(model, y_vec)

        # Verify that whiten and unwhiten are each others inverse.
        α = TemporalGPs.whiten(scalar_model, y)
        @test TemporalGPs.unwhiten(scalar_model, α) ≈ y

        # Compute square roots of psd matrices for finite differencing safety.
        sqrt_Qs = map(Q->cholesky(Symmetric(Q + 1e-2I)).U, Qs)
        sqrt_Σs = map(Σ->cholesky(Symmetric(Σ)).U, Σs)

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
                scalar_model = ScalarLGSSM(LGSSM(gmm, Σs))
                return rand(MersenneTwister(123456), scalar_model)
            end,
            randn(rng, N),
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
                scalar_model = ScalarLGSSM(LGSSM(gmm, Σs))
                return logpdf(scalar_model, y)
            end,
            randn(rng),
            As, as, sqrt_Qs, Hs, hs, x.m, P_U, sqrt_Σs, y;
            atol=1e-6, rtol=1e-6,
        )

        # Verify the gradients w.r.t. whiten
        adjoint_test(
            (As, as, sqrt_Qs, Hs, hs, m, sqrt_P, sqrt_Σs, y) -> begin
                Qs = map(U->UpperTriangular(U)'UpperTriangular(U), sqrt_Qs)
                Σs = map(U->UpperTriangular(U)'UpperTriangular(U), sqrt_Σs)
                P = UpperTriangular(sqrt_P)'UpperTriangular(sqrt_P)
                x = Gaussian(m, P)
                gmm = GaussMarkovModel(As, as, Qs, Hs, hs, x)
                scalar_model = ScalarLGSSM(LGSSM(gmm, Σs))
                return TemporalGPs.whiten(scalar_model, y)
            end,
            randn(rng, N),
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
                scalar_model = ScalarLGSSM(LGSSM(gmm, Σs))
                return TemporalGPs.unwhiten(scalar_model, α)
            end,
            randn(rng, N),
            As, as, sqrt_Qs, Hs, hs, x.m, P_U, sqrt_Σs, y;
            atol=1e-6, rtol=1e-6,
        )
    end
end
