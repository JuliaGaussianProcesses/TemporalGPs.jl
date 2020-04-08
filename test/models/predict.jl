using TemporalGPs: StaticStorage, predict

naive_predict(mf, Pf, A, a, Q) = A * mf + a, (A * Pf) * A' + Q

println("predict:")
@testset "predict" begin

    @testset "StaticArrays" begin
        rng = MersenneTwister(123456)
        Dlats = [1, 3]
        Ts = [
            (T=Float32, atol=1e-2, rtol=1e-2),
            (T=Float64, atol=1e-9, rtol=1e-9),
        ]

        @testset "$Dlat, $(T.T)" for Dlat in Dlats, T in Ts

            # Construct a Gauss-Markov model and pull out the relevant paramters.
            gmm = random_tv_gmm(rng, T.T, Dlat, 1, 1, StaticStorage())
            A = first(gmm.A)
            a = first(gmm.a)
            Q = first(gmm.Q)
            mf = gmm.x0.m
            Pf = gmm.x0.P

            # Check agreement with the naive implementation.
            mp, Pp = predict(mf, Pf, A, a, Q)
            mp_naive, Pp_naive = naive_predict(mf, Pf, A, a, Q)
            @test mp ≈ mp_naive
            @test Pp ≈ Pp_naive

            # Verify approximate numerical correctness of pullback.
            U_Pf = cholesky(Symmetric(Pf)).U
            U_Q = cholesky(Symmetric(Q)).U
            Δmp = SVector{Dlat}(randn(rng, T.T, Dlat))
            ΔPp = SMatrix{Dlat, Dlat}(randn(rng, T.T, Dlat, Dlat)) 
            adjoint_test(
                (mf, U_Pf, A, a, U_Q) -> begin
                    U_Q = UpperTriangular(U_Q)
                    U_Pf = UpperTriangular(U_Pf)                        
                    return predict(mf, U_Pf'U_Pf, A, a, U_Q'U_Q)
                end,
                (Δmp, ΔPp),
                mf, U_Pf, A, a, U_Q;
                rtol=T.rtol, atol=T.atol
            )

            # Evaluate and pullback.
            (mp, Pp), back = pullback(predict, mf, Pf, A, a, Q)
            (Δmf, ΔPf, ΔA, Δa, ΔQ) = back((Δmp, ΔPp))

            # Verify correct output types have been produced.
            @test mp isa SVector{Dlat, T.T}
            @test Pp isa SMatrix{Dlat, Dlat, T.T}

            # Verify the adjoints w.r.t. the inputs are of the correct type.
            @test Δmf isa SVector{Dlat, T.T}
            @test ΔPf isa SMatrix{Dlat, Dlat, T.T}
            @test ΔA isa SMatrix{Dlat, Dlat, T.T}
            @test Δa isa SVector{Dlat, T.T}
            @test ΔQ isa SMatrix{Dlat, Dlat, T.T}

            # Check that pullback doesn't allocate because StaticArrays.
            @test allocs(@benchmark pullback(predict, $mf, $Pf, $A, $a, $Q)) == 0
            @test allocs(@benchmark $back(($Δmp, $ΔPp))) == 0
        end
    end

    @testset "Dense" begin

        rng = MersenneTwister(123456)
        Dlats = [1, 3]
        Ts = [
            (T=Float32, atol=1e-2, rtol=1e-2),
            (T=Float64, atol=1e-9, rtol=1e-9),
        ]

        @testset "Matrix - $Dlat, $(T.T)" for Dlat in Dlats, T in Ts

            # Generate parameters for a transition model.
            A = randn(rng, T.T, Dlat, Dlat)
            a = randn(rng, T.T, Dlat)
            Q = random_nice_psd_matrix(rng, T.T, Dlat, DenseStorage())
            mf = randn(rng, T.T, Dlat)
            Pf = Symmetric(random_nice_psd_matrix(rng, T.T, Dlat, DenseStorage()))

            # Check agreement with the naive implementation.
            mp, Pp = predict(mf, Pf, A, a, Q)
            mp_naive, Pp_naive = naive_predict(mf, Pf, A, a, Q)
            @test mp ≈ mp_naive
            @test Pp ≈ Pp_naive
            @test mp isa Vector{T.T}
            @test Pp isa Matrix{T.T}

            # Verify approximate numerical correctness of pullback.
            U_Pf = cholesky(Symmetric(Pf)).U
            U_Q = cholesky(Symmetric(Q)).U
            Δmp = randn(rng, T.T, Dlat)
            ΔPp = randn(rng, T.T, Dlat, Dlat)
            adjoint_test(
                (mf, U_Pf, A, a, U_Q) -> begin
                    U_Q = UpperTriangular(U_Q)
                    U_Pf = UpperTriangular(U_Pf)
                    return predict(mf, Symmetric(U_Pf'U_Pf), A, a, U_Q'U_Q)
                end,
                (Δmp, ΔPp),
                mf, U_Pf, A, a, U_Q;
                rtol=T.rtol, atol=T.atol
            )

            # Evaluate and pullback.
            (mp, Pp), back = pullback(predict, mf, Pf, A, a, Q)
            (Δmf, ΔPf, ΔA, Δa, ΔQ) = back((Δmp, ΔPp))

            # Verify correct output types have been produced.
            @test mp isa Vector{T.T}
            @test Pp isa Matrix{T.T}

            # Verify the adjoints w.r.t. the inputs are of the correct type.
            @test Δmf isa Vector{T.T}
            @test ΔPf isa Matrix{T.T}
            @test ΔA isa Matrix{T.T}
            @test Δa isa Vector{T.T}
            @test ΔQ isa Matrix{T.T}
        end

        n_blockss = [1, 3]
        @testset "BlockDiagonal - $Dlat_block, $(T.T), $n_blocks" for
            Dlat_block in Dlats,
            T in Ts,
            n_blocks in n_blockss

            # Compute the total number of dimensions.
            Dlat = n_blocks * Dlat_block

            # Generate block-diagonal transition dynamics.
            As = map(_ -> randn(rng, T.T, Dlat_block, Dlat_block), 1:n_blocks)
            A = BlockDiagonal(As)

            a = randn(rng, T.T, Dlat)

            Qs = map(
                _ -> random_nice_psd_matrix(rng, T.T, Dlat_block, DenseStorage()),
                1:n_blocks,
            )
            Q = BlockDiagonal(Qs)

            # Generate filtering (input) distribution.
            mf = randn(rng, T.T, Dlat)
            Pf = Symmetric(random_nice_psd_matrix(rng, T.T, Dlat, DenseStorage()))

            # Check that predicting twice gives exactly the same answer.
            let
                mf_c = copy(mf)
                Pf_c = copy(Pf)
                A_c = BlockDiagonal(map(copy, As))
                a_c = copy(a)
                Q_c = BlockDiagonal(map(copy, Qs))

                m1, P1 = predict(mf_c, Pf_c, A_c, a_c, Q_c)
                m2, P2 = predict(mf_c, Pf_c, A_c, a_c, Q_c)

                @test m1 == m2
                @test P1 == P2

                @test mf_c == mf
                @test Pf_c == Pf
                @test A_c == A
                @test a_c == a
                @test Q_c == Q
            end

            # Generate corresponding dense dynamics.
            A_dense = collect(A)
            Q_dense = collect(Q)

            # Check agreement with dense implementation.
            mp, Pp = predict(mf, Pf, A, a, Q)
            mp_dense_dynamics, Pp_dense_dynamics = predict(mf, Pf, A_dense, a, Q_dense)
            @test mp ≈ mp_dense_dynamics
            @test Symmetric(Pp) ≈ Symmetric(Pp_dense_dynamics)
            @test mp isa Vector{T.T}
            @test Pp isa Matrix{T.T}

            # Verify approximate numerical correctness of pullback.
            U_Pf = collect(cholesky(Symmetric(Pf)).U)
            U_Q = map(Q -> collect(cholesky(Symmetric(Q)).U), Qs)
            Δmp = randn(rng, T.T, Dlat)
            ΔPp = randn(rng, T.T, Dlat, Dlat)

            adjoint_test(
                (mf, U_Pf, A, a, U_Q) -> begin
                    Qs = map(U -> UpperTriangular(U)'UpperTriangular(U), U_Q)
                    Q = BlockDiagonal(Qs)
                    U_Pf = UpperTriangular(U_Pf)
                    return predict(mf, Symmetric(U_Pf'U_Pf), A, a, Q)
                end,
                (Δmp, ΔPp),
                mf, U_Pf, A, a, U_Q;
                rtol=T.rtol, atol=T.atol,
            )
        end

        Ns = [1, 2]
        Ds = [2, 3]

        @testset "KroneckerProduct - $N, $D, $(T.T)" for N in Ns, D in Ds, T in Ts

            # Compute the total number of dimensions.
            Dlat = N * D

            # Generate Kronecker-Product transition dynamics.
            A_D = randn(rng, T.T, D, D)
            A = Eye{T.T}(N) ⊗ A_D

            a = randn(rng, T.T, Dlat)

            K_N = random_nice_psd_matrix(rng, T.T, N, DenseStorage())
            Q_D = random_nice_psd_matrix(rng, T.T, D, DenseStorage())
            Q = collect(K_N ⊗ Q_D)

            # Generate filtering (input) distribution.
            mf = randn(rng, T.T, Dlat)
            Pf = Symmetric(random_nice_psd_matrix(rng, T.T, Dlat, DenseStorage()))

            # Generate corresponding dense dynamics.
            A_dense = collect(A)

            # Check agreement with dense implementation.
            mp, Pp = predict(mf, Pf, A, a, Q)
            mp_dense_dynamics, Pp_dense_dynamics = predict(mf, Pf, A_dense, a, Q)
            @test mp ≈ mp_dense_dynamics
            @test Symmetric(Pp) ≈ Symmetric(Pp_dense_dynamics)
            @test mp isa Vector{T.T}
            @test Pp isa Matrix{T.T}

            # Check that predicting twice gives exactly the same answer.
            let
                mf_c = copy(mf)
                Pf_c = copy(Pf)
                A_D_c = copy(A_D)
                A_c = Eye(N) ⊗ A_D
                a_c = copy(a)
                Q_c = copy(Q)

                m1, P1 = predict(mf_c, Pf_c, A_c, a_c, Q_c)
                m2, P2 = predict(mf_c, Pf_c, A_c, a_c, Q_c)

                @test m1 == m2
                @test P1 == P2

                @test mf_c == mf
                @test Pf_c == Pf
                @test A_c == A
                @test a_c == a
                @test Q_c == Q

                (m3, P3), back = Zygote.pullback(predict, mf_c, Pf_c, A_c, a_c, Q_c)
                @test m1 == m3
                @test P1 == P3

                back((m3, P3))

                @test mf_c == mf
                @test Pf_c == Pf
                @test A_c == A
                @test a_c == a
                @test Q_c == Q
            end

            # Verify approximate numerical correctness of pullback.
            U_Pf = collect(cholesky(Symmetric(Pf)).U)
            U_Q = collect(cholesky(Symmetric(Q)).U)
            Δmp = randn(rng, T.T, Dlat)
            ΔPp = randn(rng, T.T, Dlat, Dlat)

            adjoint_test(
                (mf, U_Pf, A_D, a, U_Q) -> begin
                    U_Q = UpperTriangular(U_Q)
                    Q = collect(Symmetric(U_Q'U_Q))
                    U_Pf = UpperTriangular(U_Pf)
                    A = Eye{T.T}(N) ⊗ A_D
                    return predict(mf, Symmetric(U_Pf'U_Pf), A, a, Q)
                end,
                (Δmp, ΔPp),
                mf, U_Pf, A_D, a, U_Q;
                rtol=T.rtol, atol=T.atol,
            )
        end
    end
end
