using TemporalGPs: posterior_and_lml, predict, predict_marginals

@testset "linear_gaussian_conditionals" begin
    Dlats = [1, 3]
    Dobss = [1, 2]
    Dlats = [3]
    Dobss = [2]
    storages = [
        (name="dense storage Float64", val=ArrayStorage(Float64)),
        # (name="static storage Float64", val=SArrayStorage(Float64)),
    ]

    @testset "SmallOutputLGC (Dlat=$Dlat, Dobs=$Dobs, $(storage.name))" for
        Dlat in Dlats,
        Dobs in Dobss,
        storage in storages

        println("SmallOutputLGC (Dlat=$Dlat, Dobs=$Dobs, $(storage.name))")

        rng = MersenneTwister(123456)
        x = random_gaussian(rng, Dlat, storage.val)
        model = random_small_output_lgc(rng, Dlat, Dobs, storage.val)

        test_interface(
            rng, model, x;
            check_adjoints=true,
            check_infers=true,
            check_allocs=storage.val isa SArrayStorage,
        )

        @testset "missing data" begin

            # Generate observation with a missing.
            y = conditional_rand(rng, model, rand(rng, x))
            y_missing = Vector{Union{Missing, eltype(y)}}(undef, length(y))
            y_missing .= y
            y_missing[1] = missing

            # Create a version of the model in which the observation are `Diagonal`.
            diag_model = SmallOutputLGC(model.A, model.a, Diagonal(model.Q))

            # Compute logpdf and posterior under model given the missing.
            x_post, lml = posterior_and_lml(x, diag_model, y_missing)

            # Modify the model and compute the logpdf and posterior.
            new_model = SmallOutputLGC(
                diag_model.A[2:end, :], diag_model.a[2:end], diag_model.Q[2:end, 2:end],
            )
            y_new = y[2:end]
            x_post_new, lml_new = posterior_and_lml(x, new_model, y_new)

            # Verify that both things give the same answer.
            @test x_post ≈ x_post_new
            @test lml ≈ lml_new

            # Check that everything infers and AD gives the right answer.
            @inferred posterior_and_lml(x, diag_model, y_missing)
            adjoint_test(posterior_and_lml, (x, diag_model, y_missing))
        end
    end

    @testset "LargeOutputLGC (Dlat=$Dlat, Dobs=$Dobs, $(storage.name))" for
        Dlat in Dlats,
        Dobs in Dobss,
        storage in storages

        println("LargeOutputLGC (Dlat=$Dlat, Dobs=$Dobs, $(storage.name))")

        rng = MersenneTwister(123456)
        x = random_gaussian(rng, Dlat, storage.val)
        model = random_large_output_lgc(rng, Dlat, Dobs, storage.val)

        @testset "consistency with SmallOutputLGC" begin
            y = rand(rng, TemporalGPs.predict(x, model))
            vanilla_model = TemporalGPs.SmallOutputLGC(model.A, model.a, model.Q)
            x_vanilla, lml_vanilla = posterior_and_lml(x, vanilla_model, y)
            x_large, lml_large = posterior_and_lml(x, model, y)

            @test x_vanilla.m ≈ x_large.m
            @test x_vanilla.P ≈ x_large.P
            @test lml_vanilla ≈ lml_large

            @test predict(x, vanilla_model) ≈ predict(x, model)

            @testset "missing data" begin

                # Create missing data.
                y_missing = Vector{Union{Missing, eltype(y)}}(undef, length(y))
                y_missing .= y
                y_missing[1] = missing

                # Construct version of model with diagonal cov. mat.
                diag_model = LargeOutputLGC(model.A, model.a, Diagonal(model.Q))
                diag_vanilla_model = SmallOutputLGC(
                    vanilla_model.A, vanilla_model.a, Diagonal(vanilla_model.Q),
                )

                # Compute posterior and lml under both SmallOutputLGC and LargeOutputLGC.
                x_post_vanilla, lml_vanilla = posterior_and_lml(
                    x, diag_vanilla_model, y_missing,
                )
                x_post_large, lml_large = posterior_and_lml(x, diag_model, y_missing)

                # Check that they give roughly the same answer.
                @test x_post_vanilla ≈ x_post_large
                @test lml_vanilla ≈ lml_large

                # Check that everything infers and AD gives the right answer.
                @inferred posterior_and_lml(x, diag_model, y_missing)
                adjoint_test(posterior_and_lml, (x, diag_model, y_missing))
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

    @testset "BottleneckLGC (Din=$Din, Dmid=$Dmid, Dout=$Dout)" for
        Din in Dlats,
        Dout in Dobss,
        Dmid in Dmids

        println("BottleneckLGC (Din=$Din, Dmid=$Dmid, Dout=$Dout)")

        storage = ArrayStorage(Float64)
        rng = MersenneTwister(123456)
        x = random_gaussian(rng, Din, storage)
        model = random_bottleneck_lgc(rng, Din, Dmid, Dout, storage)

        @test TemporalGPs.dim_out(model) == Dout
        @test TemporalGPs.dim_in(model) == Din

        test_interface(
            rng, model, x;
            check_adjoints=true,
            check_infers=true,
            check_allocs=false,
        )

        @testset "consistency with SmallOutputLGC" begin
            vanilla_model = small_output_lgc_from_bottleneck(model)

            @test predict(x, vanilla_model) ≈ predict(x, model)
            @test predict_marginals(x, vanilla_model) ≈ predict_marginals(x, model)

            y = rand(rng, predict(x, model))
            x_vanilla, lml_vanilla = posterior_and_lml(x, vanilla_model, y)
            x_bottle, lml_bottle = posterior_and_lml(x, model, y)
            @test x_vanilla.P ≈ x_bottle.P rtol=1e-6
            @test x_vanilla.m ≈ x_bottle.m
            @test lml_vanilla ≈ lml_bottle

            @testset "missing data" begin

                # Create missing data.
                y_missing = Vector{Union{Missing, eltype(y)}}(undef, length(y))
                y_missing .= y
                y_missing[1] = missing

                # Construct version of model with diagonal cov. mat.
                diag_model = BottleneckLGC(
                    model.H,
                    model.h,
                    LargeOutputLGC(
                        model.fan_out.A,
                        model.fan_out.a,
                        Diagonal(model.fan_out.Q),
                    )
                )
                diag_vanilla_model = SmallOutputLGC(
                    vanilla_model.A, vanilla_model.a, Diagonal(vanilla_model.Q),
                )

                # Compute posterior and lml under both SmallOutputLGC and LargeOutputLGC.
                x_post_vanilla, lml_vanilla = posterior_and_lml(
                    x, diag_vanilla_model, y_missing,
                )
                x_post_large, lml_large = posterior_and_lml(x, diag_model, y_missing)

                # Check that they give roughly the same answer.
                @test x_post_vanilla ≈ x_post_large
                @test lml_vanilla ≈ lml_large

                # Check that everything infers and AD gives the right answer.
                @inferred posterior_and_lml(x, diag_model, y_missing)
                adjoint_test(posterior_and_lml, (x, diag_model, y_missing))
            end
        end
    end
end
