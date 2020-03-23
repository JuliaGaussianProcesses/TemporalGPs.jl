using TemporalGPs: cholesky_pullback, logdet_pullback, _predict, update_correlate,
    update_decorrelate, step_correlate, step_decorrelate, Gaussian

function _verify_pullback(f, xs, Δy, T)

    # Evaluate and pullback.
    y, back = pullback(f, xs...)
    Δxs = back(Δy)

    # Verify correct output types have been produced.
    for n in 1:length(y)
        @test y[n] isa T || y[n] isa Real
    end

    # Verify the adjoints w.r.t. the inputs are of the correct type.
    for n in 1:length(Δxs)
        @test Δxs[n] isa T
    end

    # Check that StaticArrays produce no allocations.
    if T == StaticArray
        @test allocs(@benchmark pullback($f, $xs...)) == 0
        @test allocs(@benchmark $back($Δy)) == 0
    else
        T != Array && error("Unrecognised type")
    end 
end

@testset "lgssm_pullbacks" begin
    @testset "$N" for N in [1, 2, 3]

        rng = MersenneTwister(123456)

        # Do dense stuff
        S_ = randn(rng, N, N)
        S = S_ * S_' + I
        C = cholesky(S)
        Ss = SMatrix{N, N}(S)
        Cs = cholesky(Ss)

        @testset "cholesky" begin
            C_fwd, pb = cholesky_pullback(Symmetric(S))
            Cs_fwd, pbs = cholesky_pullback(Symmetric(Ss))

            ΔC = randn(rng, N, N)
            ΔCs = SMatrix{N, N}(ΔC)

            @test C.U ≈ Cs.U
            @test Cs.U ≈ Cs_fwd.U
            @test first(pb((factors=ΔC,))) ≈ first(pbs((factors=ΔCs,)))

            @test allocs(@benchmark cholesky(Symmetric($Ss))) == 0
            @test allocs(@benchmark cholesky_pullback(Symmetric($Ss))) == 0
            @test allocs(@benchmark $pbs((factors=$ΔCs,))) == 0
        end
        @testset "logdet" begin
            @test logdet(Cs) ≈ logdet(C)
            C_fwd, pb = logdet_pullback(C)
            Cs_fwd, pbs = logdet_pullback(Cs)

            @test logdet(Cs) ≈ Cs_fwd

            Δ = randn(rng)
            @test first(pb(Δ)).factors ≈ first(pbs(Δ)).factors

            @test allocs(@benchmark logdet($Cs)) == 0
            @test allocs(@benchmark logdet_pullback($Cs)) == 0
            @test allocs(@benchmark $pbs($Δ)) == 0
        end
    end

    @testset "step pullbacks" begin
        Dlats = [3]
        Dobss = [2]
        storages = [
            (name="dense storage", val=collect, T=Array),
            (name="static storage", val=to_static, T=StaticArray),
        ]

        @testset "storage=$(storage.name), Dlat=$Dlat, Dobs=$Dobs" for
            Dlat in Dlats,
            Dobs in Dobss,
            storage in storages

            rng = MersenneTwister(123456)
            A = storage.val(randn(rng, Dlat, Dlat))
            a = storage.val(randn(rng, Dlat))
            Q = storage.val(random_nice_psd_matrix(rng, Dlat, DenseStorage()))
            U_Q = cholesky(Q).U
            H = storage.val(randn(rng, Dobs, Dlat))
            h = storage.val(randn(rng, Dobs))
            S = storage.val(random_nice_psd_matrix(rng, Dobs, DenseStorage()))
            U_S = cholesky(S).U

            m = storage.val(randn(rng, Dlat))
            P = storage.val(random_nice_psd_matrix(rng, Dlat, DenseStorage()))
            U_P = cholesky(P).U

            α = storage.val(randn(rng, Dobs))
            y = storage.val(randn(rng, Dobs))

            @testset "predict" begin

                # Specify adjoints for outputs.
                Δmp = storage.val(randn(rng, Dlat))
                ΔPp = storage.val(randn(rng, Dlat, Dlat))

                # Verify approximate numerical correctness.
                adjoint_test(
                    (mf, U_Pf, A, a, U_Q) -> begin
                        U_Q = UpperTriangular(U_Q)
                        U_Pf = UpperTriangular(U_Pf)                        
                        return _predict(mf, Symmetric(U_Pf'U_Pf), A, a, Symmetric(U_Q'U_Q))
                    end,
                    (Δmp, ΔPp),
                    m, U_P, A, a, U_Q,
                )

                _verify_pullback(_predict, (m, P, A, a, Q), (Δmp, ΔPp), storage.T)
            end
            @testset "update_correlate" begin

                # Specify adjoints for outputs.
                Δmf = storage.val(randn(rng, Dlat))
                ΔPf = storage.val(randn(rng, Dlat, Dlat))
                Δlml = randn(rng)
                Δy = storage.val(randn(rng, Dobs))
                Δout = (Δmf, ΔPf, Δlml, Δy)

                # Check reverse-mode agrees with finite differences.
                adjoint_test(
                    (mp, U_Pp, H, h, U_S, α) -> begin
                        U_Pp = UpperTriangular(U_Pp)
                        Pp = collect(Symmetric(U_Pp'U_Pp))

                        U_S = UpperTriangular(U_S)
                        S = collect(Symmetric(U_S'U_S))

                        return update_correlate(mp, Pp, H, h, S, α)
                    end,
                    Δout,
                    m, U_P, H, h, U_S, α;
                    atol=1e-6, rtol=1e-6,
                )

                # Check that appropriate typoes are produced, and allocations are correct.
                _verify_pullback(update_correlate, (m, P, H, h, S, α), Δout, storage.T)
            end
            @testset "update_decorrelate" begin

                # Specify adjoints for outputs.
                Δmf = storage.val(randn(rng, Dlat))
                ΔPf = storage.val(randn(rng, Dlat, Dlat))
                Δlml = randn(rng)
                Δα = storage.val(randn(rng, Dobs))
                Δout = (Δmf, ΔPf, Δlml, Δα)

                # Check reverse-mode agrees with finite differences.
                adjoint_test(
                    (mp, U_Pp, H, h, U_S, y) -> begin
                        U_Pp = UpperTriangular(U_Pp)
                        Pp = collect(Symmetric(U_Pp'U_Pp))

                        U_S = UpperTriangular(U_S)
                        _S = collect(Symmetric(U_S'U_S))

                        return update_correlate(mp, Pp, H, h, _S, y)
                    end,
                    Δout,
                    m, U_P, H, h, U_S, y;
                    atol=1e-6, rtol=1e-6,
                )

                # Check that appropriate typoes are produced, and allocations are correct.
                _verify_pullback(update_decorrelate, (m, P, H, h, S, y), Δout, storage.T)
            end
            @testset "step_correlate" begin

                # Specify adjoints for outputs.
                Δlml = randn(rng)
                Δy = storage.val(randn(rng, Dobs))
                Δx = Gaussian(
                    storage.val(randn(rng, Dlat)),
                    storage.val(randn(rng, Dlat, Dlat)),
                )
                Δout = (Δlml, Δy, Δx)

                # Check that appropriate typoes are produced, and allocations are correct.
                model = (A=A, a=a, Q=Q, H=H, h=h, Σ=S)
                x = Gaussian(m, P)
                args = (model, x, α)

                # Check reverse-mode agress with finite differences.
                adjoint_test(
                    (A, a, U_Q, H, h, U_S, mf, U_Pf, α) -> begin
                        U_Q = UpperTriangular(Q)
                        U_S = UpperTriangular(U_S)
                        U_Pf = UpperTriangular(U_Pf)

                        model = (
                            A=A,
                            a=a,
                            Q=collect(Symmetric(U_Q'U_Q)),
                            H=H,
                            h=h,
                            Σ=collect(Symmetric(U_S'U_S)),
                        )
                        x = Gaussian(mf, collect(Symmetric(U_Pf'U_Pf)))
                        return step_correlate(model, x, α)
                    end,
                    Δout,
                    A, a, U_Q, H, h, U_S, m, U_P, α;
                    rtol=1e-6, atol=1e-6,
                )

                # Evaluate and pullback.
                (lml, y, x), back = pullback(step_correlate, args...)
                Δmodel, Δx, Δα = back(Δout)

                # Check that output types are correct.
                T = storage.T
                @test lml isa Real
                @test y isa T
                @test x isa Gaussian{<:T, <:T}

                # Check that adjoint types are correct.
                @test Δmodel.A isa T
                @test Δmodel.a isa T
                @test Δmodel.Q isa T
                @test Δmodel.H isa T
                @test Δmodel.h isa T
                @test Δmodel.Σ isa T
                @test Δx.m isa T
                @test Δx.P isa T
                @test Δα isa T

                # Check benchmarking.
                if T == StaticArray
                    @test allocs(@benchmark pullback(step_correlate, $args...)) == 0
                    @test allocs(@benchmark $back($Δout)) == 0
                end
            end
            @testset "step_decorrelate" begin

                # Specify adjoints for outputs.
                Δlml = randn(rng)
                Δα = storage.val(randn(rng, Dobs))
                Δx = Gaussian(
                    storage.val(randn(rng, Dlat)),
                    storage.val(randn(rng, Dlat, Dlat)),
                )
                Δout = (Δlml, Δα, Δx)

                # Check reverse-mode agress with finite differences.
                adjoint_test(
                    (A, a, U_Q, H, h, U_S, mf, U_Pf, y) -> begin
                        U_Q = UpperTriangular(Q)
                        Q = collect(Symmetric(U_Q'U_Q))

                        U_S = UpperTriangular(U_S)
                        S = collect(Symmetric(U_S'U_S))

                        U_Pf = UpperTriangular(U_Pf)
                        Pf = collect(Symmetric(U_Pf'U_Pf))

                        model = (A=A, a=a, Q=Q, H=H, h=h, Σ=S)
                        x = Gaussian(mf, Pf)
                        return step_decorrelate(model, x, y)
                    end,
                    Δout,
                    A, a, U_Q, H, h, U_S, m, U_P, y;
                    atol=1e-6, rtol=1e-6,
                )
                # Check that appropriate typoes are produced, and allocations are correct.
                model = (A=A, a=a, Q=Q, H=H, h=h, Σ=S)
                x = Gaussian(m, P)
                args = (model, x, y)

                # Evaluate and pullback.
                (lml, α, x), back_decorrelate = pullback(step_decorrelate, args...)
                Δmodel, Δx, Δy = back_decorrelate(Δout)

                # Check that output types are correct.
                T = storage.T
                @test lml isa Real
                @test α isa T
                @test x isa Gaussian{<:T, <:T}

                # Check that adjoint types are correct.
                @test Δmodel.A isa T
                @test Δmodel.a isa T
                @test Δmodel.Q isa T
                @test Δmodel.H isa T
                @test Δmodel.h isa T
                @test Δmodel.Σ isa T
                @test Δx.m isa T
                @test Δx.P isa T
                @test Δy isa T

                if T == StaticArray
                    @test allocs(@benchmark pullback(step_decorrelate, $args...)) == 0
                    @test allocs(@benchmark $back_decorrelate($Δout)) == 0
                end
            end
        end
    end
end
