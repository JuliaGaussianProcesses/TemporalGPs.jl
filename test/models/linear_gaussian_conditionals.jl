using TemporalGPs: posterior_and_lml

@testset "linear_gaussian_conditionals" begin
    Dlats = [1, 3]
    Dobss = [1, 2]
    # Dlats = [3]
    # Dobss = [2]
    storages = [
        (name="dense storage Float64", val=ArrayStorage(Float64)),
        (name="static storage Float64", val=SArrayStorage(Float64)),
    ]

    @testset "SmallOutputLGC (Dlat=$Dlat, Dobs=$Dobs, $(storage.name))" for
        Dlat in Dlats,
        Dobs in Dobss,
        storage in storages

        rng = MersenneTwister(123456)
        x = random_gaussian(rng, Dlat, storage.val)
        model = random_small_output_lgc(rng, Dlat, Dobs, storage.val)

        test_interface(
            rng, model, x;
            check_adjoints=true,
            check_infers=true,
            check_allocs=storage.val isa SArrayStorage,
        )
    end

    @testset "LargeOutputLGC (Dlat=$Dlat, Dobs=$Dobs, $(storage.name))" for
        Dlat in Dlats,
        Dobs in Dobss,
        storage in storages

        rng = MersenneTwister(123456)
        x = random_gaussian(rng, Dlat, storage.val)
        model = random_large_output_lgc(rng, Dlat, Dobs, storage.val)
        y = rand(rng, TemporalGPs.predict(x, model))

        @testset "consistency with LGC" begin
            vanilla_model = TemporalGPs.SmallOutputLGC(model.A, model.a, model.Q)
            x_vanilla, lml_vanilla = posterior_and_lml(x, vanilla_model, y)
            x_large, lml_large = posterior_and_lml(x, model, y)

            @test x_vanilla.m ≈ x_large.m
            @test x_vanilla.P ≈ x_large.P
            @test lml_vanilla ≈ lml_large

            @test predict(x, vanilla_model) ≈ predict(x, model)
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
        storage in storages

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
end
