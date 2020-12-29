using TemporalGPs:
    is_of_storage_type,
    is_time_invariant,
    step_marginals,
    step_decorrelate,
    step_correlate

println("scalar_lgssm:")
@testset "scalar_lgssm" begin

    rng = MersenneTwister(123456)
    N = 3

    Ns = [1, 1001]
    Dlats = [1, 3]
    tvs = [true, false]
    storages = [
        (name="dense storage", val=ArrayStorage(Float64)),
        (name="static storage", val=SArrayStorage(Float64)),
    ]
    # Ns = [1001]
    # Dlats = [3]
    # tvs = [true]

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
        y_vec = rand(rng, model)
        y = only.(y_vec)

        @testset "steps" begin
            adjoint_test(step_marginals, (model.gmm.x0, scalar_model[1]))
            adjoint_test(step_decorrelate, (model.gmm.x0, (scalar_model[1], y[1])))
            adjoint_test(step_correlate, (model.gmm.x0, (scalar_model[1], y[1])))
        end

        ssm_interface_tests(
            rng, scalar_model;
            # check_allocs=false,
            check_allocs=storage.val isa SArrayStorage,
            max_primal_allocs=3,
            max_forward_allocs=10,
            max_backward_allocs=20,
        )

        # Compute the log marginal likelihood of the observation.
        @test logpdf(scalar_model, y) == logpdf(model, y_vec)
    end
end
