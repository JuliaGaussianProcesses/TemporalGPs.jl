using TemporalGPs:
    NoContext,
    predict,
    update_decorrelate,
    update_correlate,
    step_decorrelate,
    step_correlate,
    decorrelate,
    correlate
using Zygote: _pullback

println("immutable inference:")

@testset "immutable_inference" begin
    rng = MersenneTwister(123456)
    Dlats = [1, 3]
    Dobss = [1, 2]
    # Dlats = [3]
    # Dobss = [2]
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
        A = first(gmm.A)
        a = first(gmm.a)
        Q = first(gmm.Q)
        mf = gmm.x0.m
        Pf = gmm.x0.P
        mp = mf
        Pp = Pf

        # Verify approximate numerical correctness of pullback.
        @testset "predict pullback" begin
            Δinput = adjoint_test(predict, (mf, Pf, A, a, Q))
            @test is_of_storage_type(Δinput, storage.val)
        end

        if storage.val isa SArrayStorage
            @testset "predict doesn't allocate" begin
                check_adjoint_allocations(predict, (mf, Pf, A, a, Q))
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

        x = gmm.x0
        gmm_1 = gmm[1]

        @testset "$name performance" for (name, f, update_f, step_f) in [
            (:decorrelate, decorrelate, update_decorrelate, step_decorrelate),
            (:correlate, correlate, update_correlate, step_correlate),
        ]
            kwargs = (context=NoContext(), atol=1e-6, rtol=1e-6)
            @testset "update_$name AD correctness and inference" begin
                Δinput = adjoint_test(update_f, (mp, Pp, H, h, Σ, y); kwargs...)
                @test is_of_storage_type(Δinput, storage.val)
            end

            @testset "step_$name AD correctness and inference" begin
                Δinput = adjoint_test(step_f, ((gmm=gmm_1, Σ=ssm.Σ[1]), x, y); kwargs...)
                @test is_of_storage_type(Δinput, storage.val)
            end

            @testset "$name correctness and inference" begin
                Δinput = adjoint_test(f, (ssm, ys); kwargs...)
                @test is_of_storage_type(Δinput, storage.val)
            end

            if storage.val isa SArrayStorage
                @testset "update_$name doesn't allocate" begin
                    check_adjoint_allocations(
                        update_f, (mp, Pp, H, h, Σ, y); context=NoContext(),
                    )
                end
                @testset "step_$name doesn't allocate" begin
                    check_adjoint_allocations(
                        step_f, ((gmm=ssm.gmm[1], Σ=ssm.Σ[1]), x, y); context=NoContext(),
                    )
                end

                # These tests should pick up on any substantial changes in allocations. It's
                # possible that they'll need to be modified in future / for different
                # versions of Julia.
                @testset "$name allocations are independent of length" begin
                    @test allocs(
                        @benchmark($f($ssm, $ys); samples=1, evals=1),
                    ) <= 5
                    check_adjoint_allocations(
                        f, (ssm, ys);
                        context=NoContext(), max_forward_allocs=10, max_backward_allocs=20,
                    )
                end
            end

            # @testset "benchmarking $name" begin
            #     @show Dlat, Dobs, name
            #     _, pb = _pullback(NoContext(), f, ssm, ys)

            #     display(@benchmark($f($ssm, $ys)))
            #     println()
            #     display(@benchmark(
            #         _pullback(NoContext(), $f, $ssm, $ys),
            #     ))
            #     println()
            #     display(@benchmark($pb((randn(), $αs))))
            #     println()
            # end
        end
    end
end
