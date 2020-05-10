using TemporalGPs:
    is_of_storage_type,
    Gaussian,
    cholesky_pullback,
    logdet_pullback,
    update_correlate,
    update_correlate_pullback,
    step_correlate,
    step_correlate_pullback,
    correlate,
    correlate_pullback,
    update_decorrelate,
    update_decorrelate_pullback,
    step_decorrelate,
    step_decorrelate_pullback,
    decorrelate,
    decorrelate_pullback

naive_predict(mf, Pf, A, a, Q) = A * mf + a, (A * Pf) * A' + Q

using InteractiveUtils
function verify_pullback(f_pullback, input, Δoutput, storage)
    output, _pb = f_pullback(input...)
    Δinput = _pb(Δoutput)

    @test is_of_storage_type(input, storage.val)
    @test is_of_storage_type(output, storage.val)
    @test is_of_storage_type(Δinput, storage.val)
    @test is_of_storage_type(Δoutput, storage.val)

    if storage.val isa SArrayStorage
        @test allocs(@benchmark $f_pullback($input...)) == 0
        @test allocs(@benchmark $_pb($Δoutput)) == 0
    end
end

@testset "immutable_inference_pullbacks" begin
    @testset "$N, $T" for N in [1, 2, 3], T in [Float32, Float64]

        rng = MersenneTwister(123456)

        # Do dense stuff.
        S_ = randn(rng, T, N, N)
        S = S_ * S_' + I
        C = cholesky(S)
        Ss = SMatrix{N, N, T}(S)
        Cs = cholesky(Ss)

        @testset "cholesky" begin
            C_fwd, pb = cholesky_pullback(Symmetric(S))
            Cs_fwd, pbs = cholesky_pullback(Symmetric(Ss))

            @test eltype(C_fwd) == T
            @test eltype(Cs_fwd) == T

            ΔC = randn(rng, T, N, N)
            ΔCs = SMatrix{N, N, T}(ΔC)

            @test C.U ≈ Cs.U
            @test Cs.U ≈ Cs_fwd.U

            ΔS, = pb((factors=ΔC, ))
            ΔSs, = pbs((factors=ΔCs, ))

            @test ΔS ≈ ΔSs
            @test eltype(ΔS) == T
            @test eltype(ΔSs) == T

            @test allocs(@benchmark cholesky(Symmetric($Ss))) == 0
            @test allocs(@benchmark cholesky_pullback(Symmetric($Ss))) == 0
            @test allocs(@benchmark $pbs((factors=$ΔCs,))) == 0
        end
        @testset "logdet" begin
            @test logdet(Cs) ≈ logdet(C)
            C_fwd, pb = logdet_pullback(C)
            Cs_fwd, pbs = logdet_pullback(Cs)

            @test eltype(C_fwd) == T
            @test eltype(Cs_fwd) == T

            @test logdet(Cs) ≈ Cs_fwd

            Δ = randn(rng, T)
            ΔC = first(pb(Δ)).factors
            ΔCs = first(pbs(Δ)).factors

            @test ΔC ≈ ΔCs
            @test eltype(ΔC) == T
            @test eltype(ΔCs) == T

            @test allocs(@benchmark logdet($Cs)) == 0
            @test allocs(@benchmark logdet_pullback($Cs)) == 0
            @test allocs(@benchmark $pbs($Δ)) == 0
        end
    end

    @testset "step pullbacks" begin
        Dlats = [3]
        Dobss = [2]
        storages = [
            (name="heap - Float64", val=ArrayStorage(Float64)),
            (name="stack - Float64", val=SArrayStorage(Float64)),
            (name="heap - Float32", val=ArrayStorage(Float32)),
            (name="stack - Float32", val=SArrayStorage(Float32)),
        ]

        @testset "storage=$(storage.name), Dlat=$Dlat, Dobs=$Dobs" for
            Dlat in Dlats,
            Dobs in Dobss,
            storage in storages

            rng = MersenneTwister(123456)

            # Specify LGSSM dynamics.
            A = random_matrix(rng, Dlat, Dlat, storage.val)
            a = random_vector(rng, Dlat, storage.val)
            Q = random_nice_psd_matrix(rng, Dlat, storage.val)
            U_Q = cholesky(Q).U
            H = random_matrix(rng, Dobs, Dlat, storage.val)
            h = random_vector(rng, Dobs, storage.val)
            S = random_nice_psd_matrix(rng, Dobs, storage.val)
            U_S = cholesky(S).U

            # Specify LGSSM initial state distribution.
            m = random_vector(rng, Dlat, storage.val)
            P = random_nice_psd_matrix(rng, Dlat, storage.val)
            P = P isa Matrix ? Symmetric(P) : P
            U_P = cholesky(P).U

            # Specify input-output pairs.
            α = random_vector(rng, Dobs, storage.val)
            y = random_vector(rng, Dobs, storage.val)

            @testset "update_correlate" begin

                # Specify adjoints for outputs.
                Δmf = random_vector(rng, Dlat, storage.val)
                ΔPf = random_matrix(rng, Dlat, Dlat, storage.val)
                Δlml = randn(rng, eltype(storage.val))
                Δy = random_vector(rng, Dobs, storage.val)
                Δoutput = (Δmf, ΔPf, Δlml, Δy)

                # Check reverse-mode agrees with finite differences.
                if eltype(storage.val) == Float64
                    adjoint_test(
                        (mp, U_Pp, H, h, U_S, α) -> begin
                            U_Pp = UpperTriangular(U_Pp)
                            Pp = collect(Symmetric(U_Pp'U_Pp))

                            U_S = UpperTriangular(U_S)
                            S = collect(Symmetric(U_S'U_S))

                            return update_correlate(mp, Pp, H, h, S, α)
                        end,
                        Δoutput,
                        m, U_P, H, h, U_S, α;
                        atol=1e-6, rtol=1e-6,
                    )
                end

                # Check that appropriate typoes are produced, and allocations are correct.
                input = (m, P, H, h, S, α)
                verify_pullback(update_correlate_pullback, input, Δoutput, storage)
            end
            @testset "update_decorrelate" begin

                # Specify adjoints for outputs.
                Δmf = random_vector(rng, Dlat, storage.val)
                ΔPf = random_matrix(rng, Dlat, Dlat, storage.val)
                Δlml = randn(rng, eltype(storage.val))
                Δα = random_vector(rng, Dobs, storage.val)
                Δoutput = (Δmf, ΔPf, Δlml, Δα)
                

                # Check reverse-mode agrees with finite differences.
                if eltype(storage.val) == Float64
                    adjoint_test(
                        (mp, U_Pp, H, h, U_S, y) -> begin
                            U_Pp = UpperTriangular(U_Pp)
                            Pp = collect(Symmetric(U_Pp'U_Pp))

                            U_S = UpperTriangular(U_S)
                            _S = collect(Symmetric(U_S'U_S))

                            return update_decorrelate(mp, Pp, H, h, _S, y)
                        end,
                        Δoutput,
                        m, U_P, H, h, U_S, y;
                        atol=1e-6, rtol=1e-6,
                    )
                end

                # Check that appropriate typoes are produced, and allocations are correct.
                input = (m, P, H, h, S, y)
                verify_pullback(update_decorrelate_pullback, input, Δoutput, storage)
            end
            @testset "step_correlate" begin

                # Specify adjoints for outputs.
                Δlml = randn(rng, eltype(storage.val))
                Δy = random_vector(rng, Dobs, storage.val)
                Δx = (
                    m = random_vector(rng, Dlat, storage.val),
                    P = random_matrix(rng, Dlat, Dlat, storage.val),
                )
                Δoutput = (Δlml, Δy, Δx)

                # Check reverse-mode agress with finite differences.
                if eltype(storage.val) == Float64
                    adjoint_test(
                        (A, a, U_Q, H, h, U_S, mf, U_Pf, α) -> begin
                            U_Q = UpperTriangular(Q)
                            U_S = UpperTriangular(U_S)
                            U_Pf = UpperTriangular(U_Pf)

                            model = (
                                gmm = (
                                    A=A,
                                    a=a,
                                    Q=collect(Symmetric(U_Q'U_Q)),
                                    H=H,
                                    h=h,
                                ),
                                Σ=collect(Symmetric(U_S'U_S)),
                            )
                            x_ = Gaussian(mf, collect(Symmetric(U_Pf'U_Pf)))
                            return step_correlate(model, x_, α)
                        end,
                        Δoutput,
                        A, a, U_Q, H, h, U_S, m, U_P, α;
                        rtol=1e-6, atol=1e-6,
                    )
                end

                # Check that appropriate typoes are produced, and allocations are correct.
                model = (gmm=(A=A, a=a, Q=Q, H=H, h=h), Σ=S)
                x = Gaussian(m, P)
                input = (model, x, α)
                verify_pullback(step_correlate_pullback, input, Δoutput, storage)
            end
            @testset "step_decorrelate" begin

                # Specify adjoints for outputs.
                Δlml = randn(rng, eltype(storage.val))
                Δα = random_vector(rng, Dobs, storage.val)
                Δx = (
                    m = random_vector(rng, Dlat, storage.val),
                    P = random_matrix(rng, Dlat, Dlat, storage.val),
                )
                Δoutput = (Δlml, Δα, Δx)

                # Check reverse-mode agress with finite differences.
                if eltype(storage.val) == Float64
                    adjoint_test(
                        (A, a, U_Q, H, h, U_S, mf, U_Pf, y) -> begin
                            U_Q = UpperTriangular(Q)
                            Q = collect(Symmetric(U_Q'U_Q))

                            U_S = UpperTriangular(U_S)
                            S = collect(Symmetric(U_S'U_S))

                            U_Pf = UpperTriangular(U_Pf)
                            Pf = collect(Symmetric(U_Pf'U_Pf))

                            model = (gmm=(A=A, a=a, Q=Q, H=H, h=h), Σ=S)
                            x = Gaussian(mf, Pf)
                            return step_decorrelate(model, x, y)
                        end,
                        Δoutput,
                        A, a, U_Q, H, h, U_S, m, U_P, y;
                        atol=1e-6, rtol=1e-6,
                    )
                end

                # Check that appropriate typoes are produced, and allocations are correct.
                model = (gmm=(A=A, a=a, Q=Q, H=H, h=h), Σ=S)
                x = Gaussian(m, P)
                input = (model, x, y)
                verify_pullback(step_decorrelate_pullback, input, Δoutput, storage)
            end
        end
    end

    Dlats = [3]
    Dobss = [2]
    storages = [
        (name="heap - Float64", val=ArrayStorage(Float64)),
        (name="stack - Float64", val=SArrayStorage(Float64)),
        (name="heap - Float32", val=ArrayStorage(Float32)),
        (name="stack - Float32", val=SArrayStorage(Float32)),
    ]
    tvs = [
        (name = "time-varying", build_model = random_tv_lgssm),
        (name = "time-invariant", build_model = random_ti_lgssm),
    ]

    @testset "correlate: Dlat=$Dlat, Dobs=$Dobs, storage=$(storage.name), tv=$(tv.name)" for
        Dlat in Dlats,
        Dobs in Dobss,
        storage in storages,
        tv in tvs

        N_correctness = 10
        N_performance = 1_000

        @testset "correctness" begin
            rng = MersenneTwister(123456)

            model = tv.build_model(rng, Dlat, Dobs, N_correctness, storage.val)

            # We don't care about the statistical properties of the thing that correlate
            # is applied to, just that it's the correct size / type, for which rand 
            # suffices.
            α = rand(rng, model)

            input = (model, α)
            Δoutput = (randn(rng, eltype(storage.val)), rand(rng, model))

            output, _pb = Zygote.pullback(correlate, input...)
            Δinput = _pb(Δoutput)

            @test is_of_storage_type(input, storage.val)
            @test is_of_storage_type(output, storage.val)
            @test is_of_storage_type(Δinput, storage.val)
            @test is_of_storage_type(Δoutput, storage.val)

            # Only verify accuracy with Float64s.
            if eltype(storage.val) == Float64 && storage.val isa SArrayStorage
                adjoint_test(correlate, Δoutput, input...)
            end
        end

        # Only verify performance if StaticArrays are being used.
        if storage.val isa SArrayStorage
            @testset "performance" begin

                rng = MersenneTwister(123456)
                model = tv.build_model(rng, Dlat, Dobs, N_performance, storage.val)

                α = rand(rng, model)

                input = (model, α)
                Δoutput = (randn(rng, eltype(storage.val)), rand(rng, model))

                primal, forwards, pb = adjoint_allocs(correlate, Δoutput, input...)
                @test primal < 100
                @test forwards < 100
                @test pb < 3 * N_performance
            end
        end
    end
end
