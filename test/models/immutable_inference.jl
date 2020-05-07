using TemporalGPs: predict

naive_predict(mf, Pf, A, a, Q) = A * mf + a, (A * Pf) * A' + Q

println("immutable inference:")

@testset "immutable_inference" begin
    rng = MersenneTwister(123456)
    Dlats = [1, 3]
    Ts = [
        # (T=Float32, atol=1e-5, rtol=1e-5),
        (T=Float64, atol=1e-9, rtol=1e-9),
    ]

    @testset "$Dlat, $(T.T)" for Dlat in Dlats, T in Ts

        @testset "predict" begin

            # Construct a Gauss-Markov model and pull out the relevant parameters.
            gmm = random_tv_gmm(rng, Dlat, 1, 1, SArrayStorage(T.T))
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
end
