using TemporalGPs:
    smooth,
    Gaussian,
    GaussMarkovModel,
    is_of_storage_type,
    is_time_invariant

println("scalar_lgssm:")
@testset "scalar_lgssm" begin

    rng = MersenneTwister(123456)
    N = 3

    Ns = [1, 3]
    Dlats = [1, 3, 4]
    storages = [
        (name="dense storage", val=ArrayStorage(Float64)),
        (name="static storage", val=SArrayStorage(Float64)),
    ]
    tvs = [true, false]

    @testset "(Dlat=$Dlat, $(storage.name))" for
        Dlat in Dlats,
        storage in storages,
        N in Ns,
        tv in tvs

        # Build LGSSM.
        scalar_model = tv ?
            random_tv_scalar_lgssm(rng, Dlat, N, storage.val) :
            random_tv_scalar_lgssm(rng, Dlat, N, storage.val)

        model = scalar_model.model
        gmm = model.gmm
        Σs = model.Σ

        @test is_of_storage_type(scalar_model, storage.val)
        @test is_time_invariant(scalar_model) == false

        # Generate a sample from the model.
        y = rand(MersenneTwister(123456), scalar_model)
        y_vec = rand(MersenneTwister(123456), model)
        @test y == first.(y_vec)

        ssm_interface_tests(rng, scalar_model; check_allocs=false)

        # Compute the log marginal likelihood of the observation.
        @test logpdf(scalar_model, y) == logpdf(model, y_vec)
    end
end
