using TemporalGPs: is_of_storage_type

@testset "mutable_inference" begin
    rng = MersenneTwister(123456)
    Dlats = [1, 3]
    Dobss = [1, 2]
    Ts = [
        # (T=Float32, atol=1e-5, rtol=1e-5),
        (storage=ArrayStorage(Float64), atol=1e-9, rtol=1e-9),
    ]

    @testset "Matrix - $Dlat, $Dobs, $(T.storage)" for Dlat in Dlats, Dobs in Dobss, T in Ts

        storage = T.storage

        # Generate parameters for a transition model.
        A = random_matrix(rng, Dlat, Dlat, storage)
        a = random_vector(rng, Dlat, storage)
        Q = random_nice_psd_matrix(rng, Dlat, storage)

        mf = random_vector(rng, Dlat, storage)
        Pf = random_nice_psd_matrix(rng, Dlat, storage)
        xf = Gaussian(mf, Pf)

        mp = random_vector(rng, Dlat, storage)
        Pp = random_nice_psd_matrix(rng, Dlat, storage)
        xp = Gaussian(mp, Pp)

        # Generate parameters for emission model.
        α = random_vector(rng, Dobs, storage)
        H = random_matrix(rng, Dobs, Dlat, storage)
        h = random_vector(rng, Dobs, storage)
        Σ = random_nice_psd_matrix(rng, Dobs, storage)
        y = random_vector(rng, Dobs, storage)

        model = (gmm=(A=A, a=a, Q=Q, H=H, h=h), Σ=Σ)

        @testset "predict!" begin
            mp_naive, Pp_naive = TemporalGPs.predict(mf, Pf, A, a, Q)
            xp = TemporalGPs.predict!(xp, xf, A, a, Q)

            @test xp.m ≈ mp_naive
            @test xp.P ≈ Pp_naive
            @test is_of_storage_type(xp, storage)
        end

        @testset "update_decorrelate!" begin
            mf′_naive, Pf′_naive, lml_naive, α_naive = TemporalGPs.update_decorrelate(
                mp, Pp, H, h, Σ, y,
            )

            xf′, lml, α = TemporalGPs.update_decorrelate!(α, copy(xf), xp, H, h, Σ, y)

            @test xf′.m ≈ mf′_naive
            @test xf′.P ≈ Pf′_naive
            @test is_of_storage_type(xf′, storage)
            @test α ≈ α_naive
            @test lml ≈ lml_naive
        end

        @testset "step_decorrelate!" begin
            lml_naive, α_naive, x_filter_next_naive = TemporalGPs.step_decorrelate(
                model, xp, y,
            )

            x_filter_next = random_gaussian(rng, Dlat, storage)
            lml, α, x_filter_next = TemporalGPs.step_decorrelate!(
                α, x_filter_next, xp, model, y,
            )

            @test lml_naive ≈ lml
            @test α_naive ≈ α
            @test x_filter_next_naive.m ≈ x_filter_next.m
            @test x_filter_next_naive.P ≈ x_filter_next.P
        end

        @testset "decorrelate - mutable" begin

            model_lgssm = random_ti_lgssm(rng, Dlat, Dobs, 5, storage)
            ys = rand(rng, model_lgssm)

            lml_naive, vs_naive = TemporalGPs.decorrelate(
                TemporalGPs.Immutable(), model_lgssm, ys,
            )
            lml, vs = TemporalGPs.decorrelate(TemporalGPs.Mutable(), model_lgssm, ys)

            @test lml_naive ≈ lml
            @test all(vs_naive .≈ vs)
        end

        @testset "decorrelate - mutable - scalar" begin

            model_lgssm = random_ti_scalar_lgssm(rng, Dlat, 5, storage)
            ys = rand(rng, model_lgssm)

            lml_naive, vs_naive = TemporalGPs.decorrelate(
                TemporalGPs.Immutable(), model_lgssm, ys,
            )
            lml, vs = TemporalGPs.decorrelate(TemporalGPs.Mutable(), model_lgssm, ys)

            @test lml_naive ≈ lml
            @test all(vs_naive .≈ vs)
        end

        # @testset "predict" begin

        #     # Check agreement with the naive implementation.
        #     mp, Pp = predict(mf, Pf, A, a, Q)
        #     mp_naive, Pp_naive = naive_predict(mf, Pf, A, a, Q)
        #     @test mp ≈ mp_naive
        #     @test Pp ≈ Pp_naive
        #     @test mp isa Vector{T.T}
        #     @test Pp isa Matrix{T.T}

        #     # Verify approximate numerical correctness of pullback.
        #     U_Pf = cholesky(Symmetric(Pf)).U
        #     U_Q = cholesky(Symmetric(Q)).U
        #     Δmp = randn(rng, T.T, Dlat)
        #     ΔPp = randn(rng, T.T, Dlat, Dlat)
        #     adjoint_test(
        #         (mf, U_Pf, A, a, U_Q) -> begin
        #             U_Q = UpperTriangular(U_Q)
        #             U_Pf = UpperTriangular(U_Pf)
        #             return predict(mf, Symmetric(U_Pf'U_Pf), A, a, U_Q'U_Q)
        #         end,
        #         (Δmp, ΔPp),
        #         mf, U_Pf, A, a, U_Q;
        #         rtol=T.rtol, atol=T.atol
        #     )

        #     # Evaluate and pullback.
        #     (mp, Pp), back = pullback(predict, mf, Pf, A, a, Q)
        #     (Δmf, ΔPf, ΔA, Δa, ΔQ) = back((Δmp, ΔPp))

        #     # Verify correct output types have been produced.
        #     @test mp isa Vector{T.T}
        #     @test Pp isa Matrix{T.T}

        #     # Verify the adjoints w.r.t. the inputs are of the correct type.
        #     @test Δmf isa Vector{T.T}
        #     @test ΔPf isa Matrix{T.T}
        #     @test ΔA isa Matrix{T.T}
        #     @test Δa isa Vector{T.T}
        #     @test ΔQ isa Matrix{T.T}
        # end
    end

    # n_blockss = [1, 3]
    # @testset "BlockDiagonal - $Dlat_block, $(T.T), $n_blocks" for
    #     Dlat_block in Dlats,
    #     T in Ts,
    #     n_blocks in n_blockss

    #     rng = MersenneTwister(123456)

    #     # Compute the total number of dimensions.
    #     Dlat = n_blocks * Dlat_block

    #     # Generate block-diagonal transition dynamics.
    #     As = map(_ -> randn(rng, T.T, Dlat_block, Dlat_block), 1:n_blocks)
    #     A = BlockDiagonal(As)

    #     a = randn(rng, T.T, Dlat)

    #     Qs = map(
    #         _ -> random_nice_psd_matrix(rng, Dlat_block, ArrayStorage(T.T)),
    #         1:n_blocks,
    #     )
    #     Q = BlockDiagonal(Qs)

    #     # Generate filtering (input) distribution.
    #     mf = randn(rng, T.T, Dlat)
    #     Pf = Symmetric(random_nice_psd_matrix(rng, Dlat, ArrayStorage(T.T)))

    #     # Check that predicting twice gives exactly the same answer.
    #     let
    #         mf_c = copy(mf)
    #         Pf_c = copy(Pf)
    #         A_c = BlockDiagonal(map(copy, As))
    #         a_c = copy(a)
    #         Q_c = BlockDiagonal(map(copy, Qs))

    #         m1, P1 = predict(mf_c, Pf_c, A_c, a_c, Q_c)
    #         m2, P2 = predict(mf_c, Pf_c, A_c, a_c, Q_c)

    #         @test m1 == m2
    #         @test P1 == P2

    #         @test mf_c == mf
    #         @test Pf_c == Pf
    #         @test A_c == A
    #         @test a_c == a
    #         @test Q_c == Q
    #     end

    #     # Generate corresponding dense dynamics.
    #     A_dense = collect(A)
    #     Q_dense = collect(Q)

    #     # Check agreement with dense implementation.
    #     mp, Pp = predict(mf, Pf, A, a, Q)
    #     mp_dense_dynamics, Pp_dense_dynamics = predict(mf, Pf, A_dense, a, Q_dense)
    #     @test mp ≈ mp_dense_dynamics
    #     @test Symmetric(Pp) ≈ Symmetric(Pp_dense_dynamics)
    #     @test mp isa Vector{T.T}
    #     @test Pp isa Matrix{T.T}

    #     # Verify approximate numerical correctness of pullback.
    #     U_Pf = collect(cholesky(Symmetric(Pf)).U)
    #     U_Q = map(Q -> collect(cholesky(Symmetric(Q)).U), Qs)
    #     Δmp = randn(rng, T.T, Dlat)
    #     ΔPp = randn(rng, T.T, Dlat, Dlat)

    #     adjoint_test(
    #         (mf, U_Pf, A, a, U_Q) -> begin
    #             Qs = map(U -> UpperTriangular(U)'UpperTriangular(U), U_Q)
    #             Q = BlockDiagonal(Qs)
    #             U_Pf = UpperTriangular(U_Pf)
    #             return predict(mf, Symmetric(U_Pf'U_Pf), A, a, Q)
    #         end,
    #         (Δmp, ΔPp),
    #         mf, U_Pf, A, a, U_Q;
    #         rtol=T.rtol, atol=T.atol,
    #     )
    # end

    # Ns = [1, 2]
    # Ds = [2, 3]

    # @testset "KroneckerProduct - $N, $D, $(T.T)" for N in Ns, D in Ds, T in Ts

    #     rng = MersenneTwister(123456)
    #     storage = ArrayStorage(T.T)

    #     # Compute the total number of dimensions.
    #     Dlat = N * D

    #     # Generate Kronecker-Product transition dynamics.
    #     A_D = randn(rng, T.T, D, D)
    #     A = Eye{T.T}(N) ⊗ A_D

    #     a = randn(rng, T.T, Dlat)

    #     K_N = random_nice_psd_matrix(rng, N, storage)
    #     Q_D = random_nice_psd_matrix(rng, D, storage)
    #     Q = collect(K_N ⊗ Q_D)

    #     # Generate filtering (input) distribution.
    #     mf = randn(rng, T.T, Dlat)
    #     Pf = Symmetric(random_nice_psd_matrix(rng, Dlat, storage))

    #     # Generate corresponding dense dynamics.
    #     A_dense = collect(A)

    #     # Check agreement with dense implementation.
    #     mp, Pp = predict(mf, Pf, A, a, Q)
    #     mp_dense_dynamics, Pp_dense_dynamics = predict(mf, Pf, A_dense, a, Q)
    #     @test mp ≈ mp_dense_dynamics
    #     @test Symmetric(Pp) ≈ Symmetric(Pp_dense_dynamics)
    #     @test mp isa Vector{T.T}
    #     @test Pp isa Matrix{T.T}

    #     # Check that predicting twice gives exactly the same answer.
    #     let
    #         mf_c = copy(mf)
    #         Pf_c = copy(Pf)
    #         A_D_c = copy(A_D)
    #         A_c = Eye(N) ⊗ A_D
    #         a_c = copy(a)
    #         Q_c = copy(Q)

    #         m1, P1 = predict(mf_c, Pf_c, A_c, a_c, Q_c)
    #         m2, P2 = predict(mf_c, Pf_c, A_c, a_c, Q_c)

    #         @test m1 == m2
    #         @test P1 == P2

    #         @test mf_c == mf
    #         @test Pf_c == Pf
    #         @test A_c == A
    #         @test a_c == a
    #         @test Q_c == Q

    #         (m3, P3), back = Zygote.pullback(predict, mf_c, Pf_c, A_c, a_c, Q_c)
    #         @test m1 == m3
    #         @test P1 == P3

    #         back((m3, P3))

    #         @test mf_c == mf
    #         @test Pf_c == Pf
    #         @test A_c == A
    #         @test a_c == a
    #         @test Q_c == Q
    #     end

    #     # Verify approximate numerical correctness of pullback.
    #     U_Pf = collect(cholesky(Symmetric(Pf)).U)
    #     U_Q = collect(cholesky(Symmetric(Q)).U)
    #     Δmp = randn(rng, T.T, Dlat)
    #     ΔPp = randn(rng, T.T, Dlat, Dlat)

    #     adjoint_test(
    #         (mf, U_Pf, A_D, a, U_Q) -> begin
    #             U_Q = UpperTriangular(U_Q)
    #             Q = collect(Symmetric(U_Q'U_Q))
    #             U_Pf = UpperTriangular(U_Pf)
    #             A = Eye{T.T}(N) ⊗ A_D
    #             return predict(mf, Symmetric(U_Pf'U_Pf), A, a, Q)
    #         end,
    #         (Δmp, ΔPp),
    #         mf, U_Pf, A_D, a, U_Q;
    #         rtol=T.rtol, atol=T.atol,
    #     )
    # end

    # Ns = [1, 2, 3]
    # Ds = [1, 2, 3]
    # N_blockss = [1, 2, 3]

    # @testset "BlockDiagonal of KroneckerProduct - $N, $D, $N_blocks, $(T.T)" for
    #     N in Ns,
    #     D in Ds,
    #     N_blocks in N_blockss,
    #     T in Ts

    #     rng = MersenneTwister(123456)
    #     storage = ArrayStorage(T.T)

    #     Dlat = N * D * N_blocks

    #     # Generate BlockDiagonal-KroneckerProduct transition dynamics.
    #     A_Ds = [randn(rng, T.T, D, D) for _ in 1:N_blocks]
    #     As = [Eye{T.T}(N) ⊗ A_Ds[n] for n in 1:N_blocks]
    #     A = BlockDiagonal(As)

    #     a = randn(rng, T.T, N * D * N_blocks)

    #     Qs = [random_nice_psd_matrix(rng, N * D, storage) for _ in 1:N_blocks]
    #     Q = BlockDiagonal(Qs)

    #     # Generate filtering (input) distribution.
    #     mf = randn(rng, T.T, Dlat)
    #     Pf = Symmetric(random_nice_psd_matrix(rng, Dlat, storage))

    #     # Generate corresponding dense dynamics.
    #     A_dense = collect(A)
    #     Q_dense = collect(Q)

    #     # Check agreement with dense implementation.
    #     mp, Pp = predict(mf, Pf, A, a, Q)
    #     mp_dense_dynamics, Pp_dense_dynamics = predict(mf, Pf, A_dense, a, Q_dense)
    #     @test mp ≈ mp_dense_dynamics
    #     @test Symmetric(Pp) ≈ Symmetric(Pp_dense_dynamics) atol=1e-6 rtol=1e-6

    #     @test A_dense == A
    #     @test Q_dense == Q

    #     @test mp isa Vector{T.T}
    #     @test Pp isa Matrix{T.T}

    #     # Verify approximate numerical correctness of pullback.
    #     U_Pf = collect(cholesky(Symmetric(Pf)).U)
    #     U_Q = map(Q -> collect(cholesky(Symmetric(Q)).U), Qs)
    #     Δmp = randn(rng, T.T, Dlat)
    #     ΔPp = randn(rng, T.T, Dlat, Dlat)

    #     adjoint_test(
    #         (mf, U_Pf, A_Ds, a, U_Q) -> begin
    #             Qs = map(U -> UpperTriangular(U)'UpperTriangular(U), U_Q)
    #             Q = BlockDiagonal(Qs)
    #             U_Pf = UpperTriangular(U_Pf)
    #             A = BlockDiagonal(map(A_D -> Eye{T.T}(N) ⊗ A_D, A_Ds))
    #             return predict(mf, Symmetric(U_Pf'U_Pf), A, a, Q)
    #         end,
    #         (Δmp, ΔPp),
    #         mf, U_Pf, A_Ds, a, U_Q;
    #         rtol=T.rtol, atol=T.atol,
    #     )
    # end
end
