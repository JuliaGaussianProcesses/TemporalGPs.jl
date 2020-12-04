using TemporalGPs:
    NoContext,
    predict,
    update_decorrelate,
    update_correlate,
    step_decorrelate,
    step_correlate,
    decorrelate,
    correlate,
    copy_first
using Zygote: _pullback

naive_predict(mf, Pf, A, a, Q) = A * mf + a, (A * Pf) * A' + Q

println("immutable inference:")

@testset "immutable_inference" begin
    rng = MersenneTwister(123456)
    Dlats = [1, 3]
    Dobss = [1, 2]
    Ts = [
        # (T=Float32, atol=1e-5, rtol=1e-5),
        (T=Float64, atol=1e-9, rtol=1e-9),
    ]

    @testset "$Dlat, $Dobs, $(T.T)" for Dlat in Dlats, Dobs in Dobss, T in Ts

        # Construct a Gauss-Markov model and pull out the relevant parameters.
        gmm = random_tv_gmm(rng, Dlat, Dobs, 1, SArrayStorage(T.T))
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

        @testset "predict AD infers" begin
            (mp, Pp), pb = _pullback(NoContext(), predict, mf, Pf, A, a, Q)
            @inferred _pullback(NoContext(), predict, mf, Pf, A, a, Q)
            @inferred pb((Δmp, ΔPp))
        end

        @testset "predict doesn't allocate" begin
            _, pb = _pullback(NoContext(), predict, mf, Pf, A, a, Q)
            @test allocs(@benchmark(
                _pullback(NoContext(), predict, $mf, $Pf, $A, $a, $Q),
                samples=1,
                evals=1,
            )) == 0
            @test allocs(@benchmark $pb(($Δmp, $ΔPp)) samples=1 evals=1) == 0
        end

        H = first(gmm.H)
        h = first(gmm.h)
        Σ = random_nice_psd_matrix(rng, Dobs, SArrayStorage(T.T))
        y = random_vector(rng, Dobs, SArrayStorage(T.T))

        Δmf = random_vector(rng, Dlat, SArrayStorage(T.T))
        ΔPf = random_matrix(rng, Dlat, Dlat, SArrayStorage(T.T))
        Δlml = randn(rng)
        Δα = random_vector(rng, Dobs, SArrayStorage(T.T))

        x = random_gaussian(rng, Dlat, SArrayStorage(T.T))
        lgssm = random_tv_lgssm(rng, Dlat, Dobs, 1_000, SArrayStorage(T.T))
        ys = rand(rng, lgssm)
        αs = rand(rng, lgssm)

        @testset "$name performance" for (name, f, update_f, step_f) in [
            (:decorrelate, decorrelate, update_decorrelate, step_decorrelate),
            (:correlate, correlate, update_correlate, step_correlate),
        ]

            @testset "update_$name AD infers" begin
                _, pb = _pullback(NoContext(), update_f, mp, Pp, H, h, Σ, y)
                @inferred _pullback(NoContext(), update_f, mp, Pp, H, h, Σ, y)
                @inferred pb((Δmf, ΔPf, Δlml, Δα))
            end

            @testset "update_$name doesn't allocate" begin
                _, pb = _pullback(NoContext(), update_f, mp, Pp, H, h, Σ, y)
                @test allocs(@benchmark(
                    _pullback(NoContext(), $update_f, $mp, $Pp, $H, $h, $Σ, $y),
                    samples=1,
                    evals=1,
                )) == 0
                @test allocs(@benchmark $pb(($Δmf, $ΔPf, $Δlml, $Δα)) samples=1 evals=1) == 0
            end

            @testset "step_$name AD infers" begin
                model = (gmm=lgssm.gmm[1], Σ=lgssm.Σ[1])
                Δ = (Δlml, Δα, (m=Δmf, P=ΔPf))
                out, pb = _pullback(NoContext(), step_f, model, x, y)
                @inferred _pullback(NoContext(), step_f, model, x, y)
                @inferred pb(Δ)
            end

            @testset "step_$name doesn't allocate" begin
                model = (gmm=lgssm.gmm[1], Σ=lgssm.Σ[1])
                Δ = (Δlml, Δα, (m=Δmf, P=ΔPf))
                _, pb = _pullback(NoContext(), step_f, model, x, y)
                @test allocs(@benchmark(
                    _pullback(NoContext(), $step_f, $model, $x, $y),
                    samples=1,
                    evals=1,
                )) == 0
                @test allocs(@benchmark $pb($Δ) samples=1 evals=1) == 0
            end

            @testset "$name infers" begin
                _, pb = _pullback(NoContext(), f, lgssm, ys)
                @inferred f(lgssm, ys, copy_first)
                @inferred _pullback(NoContext(), f, lgssm, ys, copy_first)
                @inferred pb((randn(), αs))
            end

            # These tests should pick up on any substantial changes in allocations. It's
            # possible that they'll need to be modified in future / for different versions
            # of Julia.
            @testset "$name allocations are independent of length" begin
                _, pb = _pullback(NoContext(), f, lgssm, ys, copy_first)

                @test allocs(
                    @benchmark($f($lgssm, $ys, copy_first); samples=1, evals=1),
                ) < 5
                @test allocs(
                    @benchmark(
                        _pullback(NoContext(), $f, $lgssm, $ys, copy_first);
                        samples=1, evals=1,
                    ),
                ) < 10
                @test allocs(@benchmark($pb((randn(), $αs)); samples=1, evals=1)) < 20
            end

            # @testset "benchmarking $name" begin
            #     @show Dlat, Dobs, name, T.T
            #     _, pb = _pullback(NoContext(), f, lgssm, ys, copy_first)

            #     display(@benchmark($f($lgssm, $ys, copy_first)))
            #     println()
            #     display(@benchmark(
            #         _pullback(NoContext(), $f, $lgssm, $ys, copy_first),
            #     ))
            #     println()
            #     display(@benchmark($pb((randn(), $αs))))
            #     println()
            # end
        end
    end
end
