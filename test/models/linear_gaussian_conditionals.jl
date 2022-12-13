using TemporalGPs: posterior_and_lml, predict, predict_marginals

println("linear_gaussian_conditionals:")
@testset "linear_gaussian_conditionals" begin
    Dlats = [1, 3]
    Dobss = [1, 2]
    # Dlats = [3]
    # Dobss = [2]
    storages = [
        (name="dense storage Float64", val=ArrayStorage(Float64)),
    ]
    Q_types = [
        Val(:dense),
        Val(:diag),
    ]

    @testset "SmallOutputLGC (Dlat=$Dlat, Dobs=$Dobs, Q=$(Q_type), $(storage.name))" for
        Dlat in Dlats,
        Dobs in Dobss,
        Q_type in Q_types,
        storage in storages

        println("SmallOutputLGC (Dlat=$Dlat, Dobs=$Dobs, Q=$(Q_type), $(storage.name))")

        rng = MersenneTwister(123456)
        x = random_gaussian(rng, Dlat, storage.val)
        model = random_small_output_lgc(rng, Dlat, Dobs, Q_type, storage.val)

        test_interface(
            rng, model, x;
            check_adjoints=true,
            check_infers=true,
            check_allocs=storage.val isa SArrayStorage,
        )

        Q_type == Val(:diag) && @testset "missing data" begin

            # Generate observation with a missing.
            y = conditional_rand(rng, model, rand(rng, x))
            y_missing = Vector{Union{Missing, eltype(y)}}(undef, length(y))
            y_missing .= y
            y_missing[1] = missing

            # Compute logpdf and posterior under model given the missing.
            x_post, lml = posterior_and_lml(x, model, y_missing)

            # Modify the model and compute the logpdf and posterior.
            new_model = SmallOutputLGC(
                model.A[2:end, :], model.a[2:end], model.Q[2:end, 2:end],
            )
            y_new = y[2:end]
            x_post_new, lml_new = posterior_and_lml(x, new_model, y_new)

            # Verify that both things give the same answer.
            @test x_post ≈ x_post_new
            @test lml ≈ lml_new atol=1e-8 rtol=1e-8

            # Check that everything infers and AD gives the right answer.
            @inferred posterior_and_lml(x, model, y_missing)
            x̄ = adjoint_test(posterior_and_lml, (x, model, y_missing))
            @test x̄[2].Q isa NamedTuple{(:diag, )}
        end
    end

    @testset "LargeOutputLGC (Dlat=$Dlat, Dobs=$Dobs, Q=$(Q_type), $(storage.name))" for
        Dlat in Dlats,
        Dobs in Dobss,
        Q_type in Q_types,
        storage in storages

        println("LargeOutputLGC (Dlat=$Dlat, Dobs=$Dobs, Q=$(Q_type), $(storage.name))")

        rng = MersenneTwister(123456)
        x = random_gaussian(rng, Dlat, storage.val)
        model = random_large_output_lgc(rng, Dlat, Dobs, Q_type, storage.val)

        @testset "consistency with SmallOutputLGC" begin
            y = rand(rng, TemporalGPs.predict(x, model))
            vanilla_model = TemporalGPs.SmallOutputLGC(model.A, model.a, model.Q)
            x_vanilla, lml_vanilla = posterior_and_lml(x, vanilla_model, y)
            x_large, lml_large = posterior_and_lml(x, model, y)

            @test x_vanilla.m ≈ x_large.m
            @test x_vanilla.P ≈ x_large.P
            @test lml_vanilla ≈ lml_large

            @test predict(x, vanilla_model) ≈ predict(x, model)

            Q_type == Val(:diag) && @testset "missing data" begin

                # Create missing data.
                y_missing = Vector{Union{Missing, eltype(y)}}(undef, length(y))
                y_missing .= y
                y_missing[1] = missing

                # Compute posterior and lml under both SmallOutputLGC and LargeOutputLGC.
                x_post_vanilla, lml_vanilla = posterior_and_lml(x, vanilla_model, y_missing)
                x_post_large, lml_large = posterior_and_lml(x, model, y_missing)

                # Check that they give roughly the same answer.
                @test x_post_vanilla ≈ x_post_large
                @test lml_vanilla ≈ lml_large rtol=1e-8 atol=1e-8

                # Check that everything infers and AD gives the right answer.
                @inferred posterior_and_lml(x, model, y_missing)
                x̄ = adjoint_test(posterior_and_lml, (x, model, y_missing))
                @test x̄[2].Q isa NamedTuple{(:diag, )}
            end
        end

        test_interface(
            rng, model, x;
            check_adjoints=true,
            check_infers=true,
            check_allocs=storage.val isa SArrayStorage,
        )
    end

    @testset "ScalarOutputLGC (Dlat=$Dlat, ($storage.name))" for
        Dlat in Dlats,
        storage in [
            (name="dense storage Float64", val=ArrayStorage(Float64)),
            (name="static storage Float64", val=SArrayStorage(Float64)),
        ]

        println("ScalarOutputLGC (Dlat=$Dlat, ($storage.name))")

        rng = MersenneTwister(123456)
        x = random_gaussian(rng, Dlat, storage.val)
        model = random_scalar_output_lgc(rng, Dlat, storage.val)

        @testset "consistency with LGC" begin
            vanilla_model = lgc_from_scalar_output_lgc(model)
            y_vanilla = rand(rng, predict(x, vanilla_model))
            x_vanilla, lml_vanilla = posterior_and_lml(x, vanilla_model, y_vanilla)
            x_scalar, lml_scalar = posterior_and_lml(x, model, only(y_vanilla))

            @test x_vanilla.m ≈ x_scalar.m
            @test x_vanilla.P ≈ x_scalar.P
            @test lml_vanilla ≈ lml_scalar
        end

        test_interface(
            rng, model, x;
            check_adjoints=true,
            check_infers=true,
            check_allocs=storage.val isa SArrayStorage,
        )
    end

    Dmids = [1, 3]

    @testset "BottleneckLGC (Din=$Din, Dmid=$Dmid, Dout=$Dout, Q=$(Q_type))" for
        Din in Dlats,
        Dout in Dobss,
        Dmid in Dmids,
        Q_type in Q_types

        println("BottleneckLGC (Din=$Din, Dmid=$Dmid, Dout=$Dout, Q=$(Q_type))")

        storage = ArrayStorage(Float64)
        rng = MersenneTwister(123456)
        x = random_gaussian(rng, Din, storage)
        model = random_bottleneck_lgc(rng, Din, Dmid, Dout, Q_type, storage)

        @test TemporalGPs.dim_out(model) == Dout
        @test TemporalGPs.dim_in(model) == Din

        test_interface(
            rng, model, x;
            check_adjoints=true,
            check_infers=TEST_TYPE_INFER,
            check_allocs=TEST_ALLOC,
        )

        @testset "consistency with SmallOutputLGC" begin
            vanilla_model = small_output_lgc_from_bottleneck(model)

            @test predict(x, vanilla_model) ≈ predict(x, model)
            @test predict_marginals(x, vanilla_model) ≈ predict_marginals(x, model)

            y = rand(rng, predict(x, model))
            x_vanilla, lml_vanilla = posterior_and_lml(x, vanilla_model, y)
            x_bottle, lml_bottle = posterior_and_lml(x, model, y)
            @test x_vanilla.P ≈ x_bottle.P rtol=1e-6
            @test x_vanilla.m ≈ x_bottle.m rtol=1e-6
            @test lml_vanilla ≈ lml_bottle rtol=1e-6

            Q_type == Val(:diag) && @testset "missing data" begin

                # Create missing data.
                y_missing = Vector{Union{Missing, eltype(y)}}(undef, length(y))
                y_missing .= y
                y_missing[1] = missing

                # Compute posterior and lml under both SmallOutputLGC and LargeOutputLGC.
                x_post_vanilla, lml_vanilla = posterior_and_lml(x, vanilla_model, y_missing)
                x_post_large, lml_large = posterior_and_lml(x, model, y_missing)

                # Check that they give roughly the same answer.
                @test x_post_vanilla ≈ x_post_large rtol=1e-8 atol=1e-8
                @test lml_vanilla ≈ lml_large rtol=1e-8 atol=1e-8

                # Check that everything infers and AD gives the right answer.
                @inferred posterior_and_lml(x, model, y_missing)
                x̄ = adjoint_test(posterior_and_lml, (x, model, y_missing))
                @test x̄[2].fan_out.Q isa NamedTuple{(:diag, )}
            end
        end
    end
end
