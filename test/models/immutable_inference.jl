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
    tvs = [
        (name = "time-varying", build_model = random_tv_lgssm),
        (name = "time-invariant", build_model = random_ti_lgssm),
    ]
    storages = [
        (name="heap - Float64", val=ArrayStorage(Float64)),
        (name="stack - Float64", val=SArrayStorage(Float64)),
        # (name="heap - Float32", val=ArrayStorage(Float32)),
        # (name="stack - Float32", val=SArrayStorage(Float32)),
    ]
    @testset "$Dlat, $Dobs, $(storage.val), $(tv.name)" for
        Dlat in Dlats,
        Dobs in Dobss,
        storage in storages,
        tv in tvs

        println("$Dlat, $Dobs, $(storage.val), $(tv.name)")

        # Construct a Gauss-Markov model and pull out the relevant parameters.
        ssm = tv.build_model(rng, Dlat, Dobs, 1_000, storage.val)
        gmm = ssm.gmm
        # gmm = random_tv_gmm(rng, Dlat, Dobs, 1, SArrayStorage(T.T))
        A = first(gmm.A)
        a = first(gmm.a)
        Q = first(gmm.Q)
        mf = gmm.x0.m
        Pf = gmm.x0.P

        # Generate tangents for the ssm.
        ∂ssm = rand_tangent(ssm)
        ∂gmm = ∂ssm.gmm
        ∂A = rand_tangent(A)
        ∂a = rand_tangent(a)
        ∂Q = random_nice_psd_matrix(rng, length(a), storage.val)
        ∂mf = rand_tangent(mf)
        ∂Pf = random_nice_psd_matrix(rng, length(mf), storage.val)

        # Check agreement with the naive implementation.
        mp, Pp = predict(mf, Pf, A, a, Q)
        mp_naive, Pp_naive = naive_predict(mf, Pf, A, a, Q)
        @test mp ≈ mp_naive
        @test Pp ≈ Pp_naive

        # Generate tangents / cotangents for mp and Pp.
        ∂mp = rand_tangent(mp)
        ∂Pp = random_nice_psd_matrix(rng, length(mp), storage.val)
        Δmp = rand_zygote_tangent(mp)
        ΔPp = rand_zygote_tangent(Pp)

        # Verify approximate numerical correctness of pullback.
        @testset "predict pullback" begin
            input = (mf, Pf, A, a, Q)
            ∂input = (∂mf, ∂Pf, ∂A, ∂a, ∂Q)
            Δoutput = (Δmp, ΔPp)
            adjoint_test(predict, Δoutput, input, ∂input)
        end

        # Evaluate and pullback.
        @testset "predict types" begin
            output, back = pullback(predict, mf, Pf, A, a, Q)
            Δinput = back((Δmp, ΔPp))
            @test is_of_storage_type(output, storage.val)
            @test is_of_storage_type(Δinput, storage.val)
        end

        if storage.val isa SArrayStorage
            @testset "predict doesn't allocate" begin
                check_adjoint_allocations(predict, (Δmp, ΔPp), mf, Pf, A, a, Q)
            end
        end

        H = first(gmm.H)
        h = first(gmm.h)
        Σ = first(ssm.Σ)
        ys = rand(ssm)
        αs = rand(ssm)

        y = first(ys)
        α = first(αs)
        lml = logpdf(ssm, ys)

        ∂H = rand_tangent(H)
        ∂h = rand_tangent(h)
        ∂Σ = random_nice_psd_matrix(rng, size(Σ, 1), storage.val)
        ∂ys = rand_tangent(ys)
        ∂αs = rand_tangent(αs)
        ∂y = rand_tangent(y)
        ∂α = rand_tangent(α)
        ∂lml = rand_tangent(lml)

        Δmf = rand_zygote_tangent(mf)
        ΔPf = rand_zygote_tangent(Pf)
        Δlml = rand_zygote_tangent(lml)
        Δα = rand_zygote_tangent(first(αs))
        Δαs = rand_zygote_tangent(αs)

        x = gmm.x0
        ∂x = rand_tangent(x)

        gmm_1 = gmm[1]
        ∂gmm_1 = Composite{typeof(gmm_1)}(A=∂A, a=∂a, Q=∂Q, H=∂H, h=∂h)

        @testset "$name performance" for (name, f, update_f, step_f) in [
            (:decorrelate, decorrelate, update_decorrelate, step_decorrelate),
            (:correlate, correlate, update_correlate, step_correlate),
        ]
            @testset "update_$name AD correctness and inference" begin
                input = (mp, Pp, H, h, Σ, y)
                ∂input = (∂mp, ∂Pp, ∂H, ∂h, ∂Σ, ∂y)
                Δoutput = (Δmf, ΔPf, Δlml, Δα)
                Δinput = adjoint_test(
                    update_f, Δoutput, input, ∂input;
                    context=NoContext(),
                    test=eltype(storage.val) == Float64,
                    atol=1e-6, rtol=1e-6,
                )
                @test is_of_storage_type(Δinput, storage.val)
            end

            @testset "step_$name AD correctness and inference" begin
                input = ((gmm=gmm_1, Σ=ssm.Σ[1]), x, y)
                ∂input = ((gmm=∂gmm_1, Σ=∂Σ), ∂x, ∂y)
                Δoutput = (Δlml, Δα, (m=Δmf, P=ΔPf))
                Δinput = adjoint_test(
                    step_f, Δoutput, input, ∂input;
                    context=NoContext(),
                    test=eltype(storage.val) == Float64,
                    atol=1e-6, rtol=1e-6,
                )
                @test is_of_storage_type(Δinput, storage.val)
            end

            @testset "$name correctness and inference, $(ot.name)" for ot in [
                (name = "copy_first", f=copy_first),
                (name = "pick_last", f=pick_last),
            ]
                input = (ssm, ys, ot.f)
                ∂input = (∂ssm, ∂ys, NO_FIELDS)
                Δoutput = rand_zygote_tangent(f(input...))
                Δinput = adjoint_test(
                    f, Δoutput, input, ∂input;
                    context=NoContext(),
                    test=eltype(storage.val) == Float64,
                    atol=1e-6, rtol=1e-6,
                )
                @test is_of_storage_type(Δinput, storage.val)
            end

            if storage.val isa SArrayStorage
                @testset "update_$name doesn't allocate" begin
                    check_adjoint_allocations(
                        update_f,
                        (Δmf, ΔPf, Δlml, Δα),
                        mp, Pp, H, h, Σ, y;
                        context=NoContext(),
                        atol=1e-6, rtol=1e-6,
                    )
                end
                @testset "step_$name doesn't allocate" begin
                    check_adjoint_allocations(
                        step_f,
                        (Δlml, Δα, (m=Δmf, P=ΔPf)),
                        (gmm=ssm.gmm[1], Σ=ssm.Σ[1]), x, y;
                        context=NoContext(),
                        atol=1e-6, rtol=1e-6,
                    )
                end

                # These tests should pick up on any substantial changes in allocations. It's
                # possible that they'll need to be modified in future / for different
                # versions of Julia.
                @testset "$name allocations are independent of length" begin
                    @test allocs(
                        @benchmark($f($ssm, $ys, copy_first); samples=1, evals=1),
                    ) <= 5
                    check_adjoint_allocations(
                        f, (Δlml, Δαs), ssm, ys, copy_first;
                        context=NoContext(), max_forward_allocs=10, max_backward_allocs=20,
                    )
                end
            end

            # @testset "benchmarking $name" begin
            #     @show Dlat, Dobs, name
            #     _, pb = _pullback(NoContext(), f, ssm, ys, copy_first)

            #     display(@benchmark($f($ssm, $ys, copy_first)))
            #     println()
            #     display(@benchmark(
            #         _pullback(NoContext(), $f, $ssm, $ys, copy_first),
            #     ))
            #     println()
            #     display(@benchmark($pb((randn(), $αs))))
            #     println()
            # end
        end
    end
end
