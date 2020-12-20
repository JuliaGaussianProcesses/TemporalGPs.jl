using TemporalGPs:
    smooth,
    Gaussian,
    GaussMarkovModel,
    is_of_storage_type,
    is_time_invariant,
    from_vector_observations,
    to_vector_observations,
    NoContext

println("scalar_lgssm:")
@testset "scalar_lgssm" begin

    # @testset "from_vector_observations pullback" begin
    #     @testset "Vector" begin
    #         y_vecs = [randn(1) for _ in 1:10]
    #         adjoint_test(from_vector_observations, (y_vecs, ))
    #         adjoint_test(
    #             x-> to_vector_observations(ArrayStorage(Float64), x), (randn(10), ),
    #         )
    #     end
    #     @testset "SVector" begin
    #         adjoint_test(from_vector_observations, ([SVector{1}(randn()) for _ in 1:11], ))
    #         adjoint_test(
    #             x-> to_vector_observations(SArrayStorage(Float64), x),
    #             (SVector{10}(randn(10)), ),
    #         )
    #     end
    # end

    rng = MersenneTwister(123456)
    N = 3

    Dlats = [1, 3, 4]
    storages = [
        (name="dense storage", val=ArrayStorage(Float64)),
        (name="static storage", val=SArrayStorage(Float64)),
    ]

    @testset "(Dlat=$Dlat, $(storage.name))" for
        Dlat in Dlats,
        storage in storages

        # Build LGSSM.
        scalar_model = random_tv_scalar_lgssm(rng, Dlat, N, storage.val)
        model = scalar_model.model
        gmm = model.gmm
        Σs = model.Σ
        As, as, Qs, Hs, hs, x = gmm.A, gmm.a, gmm.Q, gmm.H, gmm.h, gmm.x0

        @test is_of_storage_type(scalar_model, storage.val)
        @test is_time_invariant(scalar_model) == false

        # Generate a sample from the model.
        y = rand(MersenneTwister(123456), scalar_model)
        y_vec = rand(MersenneTwister(123456), model)
        @test y == first.(y_vec)

        ssm_interface_tests(rng, scalar_model; check_adjoints=true, context=NoContext())

        # Compute the log marginal likelihood of the observation.
        @test logpdf(scalar_model, y) == logpdf(model, y_vec)
    end
end
