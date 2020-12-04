using TemporalGPs: is_of_storage_type, correlate, decorrelate

@testset "immutable_inference_pullbacks" begin

    # AD correctness testing.
    fs = [
        (name="decorrelate", f=decorrelate),
        (name="correlate", f=correlate),
    ]
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

    @testset "$(f.name): Dlat=$Dlat, Dobs=$Dobs, storage=$(storage.name), tv=$(tv.name)" for
        f in fs,
        Dlat in Dlats,
        Dobs in Dobss,
        storage in storages,
        tv in tvs

        rng = MersenneTwister(123456)

        model = tv.build_model(rng, Dlat, Dobs, 10, storage.val)

        # We don't care about the statistical properties of the thing that correlate
        # is applied to, just that it's the correct size / type, for which rand
        # suffices.
        α = rand(rng, model)

        input = (model, α)
        Δoutput = (randn(rng, eltype(storage.val)), rand(rng, model))

        output, _pb = Zygote.pullback(f.f, input...)
        Δinput = _pb(Δoutput)

        @test is_of_storage_type(input, storage.val)
        @test is_of_storage_type(output, storage.val)
        @test is_of_storage_type(Δinput, storage.val)
        @test is_of_storage_type(Δoutput, storage.val)

        # Only verify accuracy with Float64s.
        if eltype(storage.val) == Float64 && storage.val isa SArrayStorage
            adjoint_test(f.f, Δoutput, input...)
        end
    end
end
