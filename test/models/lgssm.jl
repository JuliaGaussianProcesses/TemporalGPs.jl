using TemporalGPs:
    TemporalGPs,
    predict,
    step_marginals,
    step_logpdf,
    step_filter,
    step_rand,
    invert_dynamics,
    step_posterior,
    storage_type,
    is_of_storage_type,
    ArrayStorage,
    SArrayStorage,
    SmallOutputLGC,
    LargeOutputLGC,
    ScalarOutputLGC,
    Forward,
    Reverse,
    ordering

println("lgssm:")
@testset "lgssm" begin

    rng = MersenneTwister(123456)

    storages = (
        dense=(name="dense storage Float64", val=ArrayStorage(Float64)),
        static=(name="static storage Float64", val=SArrayStorage(Float64)),
    )
    emission_types = (
        small_output=(name="small output", val=SmallOutputLGC),
        large_output=(name="large output", val=LargeOutputLGC),
        scalar_output=(name="scalar output", val=ScalarOutputLGC),
    )
    settings = [
        (tv=:time_varying, N=1, Dlat=3, Dobs=2, storage=storages.dense),
        (tv=:time_varying, N=49, Dlat=3, Dobs=2, storage=storages.dense),
        # (tv=:time_invariant, N=49, Dlat=3, Dobs=2, storage=storages.dense),
        (tv=:time_varying, N=49, Dlat=1, Dobs=1, storage=storages.dense),
        (tv=:time_varying, N=1, Dlat=3, Dobs=2, storage=storages.static),
        # (tv=:time_invariant, N=49, Dlat=3, Dobs=2, storage=storages.static),
    ]
    orderings = [
        Forward(),
        Reverse(),
    ]
    Qs = [
        Val(:dense),
        # Val(:diag), diag tests don't work because `FiniteDiffernces.to_vec`.
    ]

    @testset "($tv, $N, $Dlat, $Dobs, $(storage.name), $(emission.name), $order, $Q)" for
        (tv, N, Dlat, Dobs, storage) in settings,
        emission in emission_types,
        order in orderings,
        Q in Qs

        # Print current iteration to prevent CI timing out.
        println(
            "(time_varying=$tv, N=$N, Dlat=$Dlat, Dobs=$Dobs, " *
            "storage=$(storage.name), emissions=$(emission.val), ordering=$order)",
        )

        # Build LGSSM.
        model = if emission.val ∈ (SmallOutputLGC, LargeOutputLGC)
            random_lgssm(rng, order, Val(tv), emission.val, Dlat, Dobs, N, Q, storage.val)
        elseif emission.val == ScalarOutputLGC
            random_lgssm(rng, order, Val(tv), emission.val, Dlat, Dobs, N, storage.val)
        else
            throw(error("Unrecognised storage $(emission.val)"))
        end

        # Verify the correct output types has been obtained.
        @test eltype(model.emissions) <: emission.val

        # Verify that model properties are as requested.
        @test storage_type(model) == storage.val

        @test length(model) == N
        @test getindex(model, N) isa TemporalGPs.ElementOfLGSSM

        y = first(rand(model))
        x = TemporalGPs.x0(model)

        @testset "step_marginals" begin
            @inferred step_marginals(x, model[1])
            test_rule(rng, step_marginals, x, model[1]; is_primitive=false)

        end
        @testset "step_logpdf" begin
            args = (ordering(model[1]), x, (model[1], y))
            @inferred step_logpdf(args...)
            test_rule(rng, step_logpdf, args...; is_primitive=false)
        end
        @testset "step_filter" begin
            args = (ordering(model[1]), x, (model[1], y))
            @inferred step_filter(args...)
            test_rule(rng, step_filter, args...; is_primitive=false)
        end
        @testset "invert_dynamics" begin
            args = (x, x, model[1].transition)
            @inferred invert_dynamics(args...)
            test_rule(rng, invert_dynamics, args...; is_primitive=false)
        end
        @testset "step_posterior" begin
            args = (ordering(model[1]), x, (model[1], y))
            @inferred step_posterior(args...)
            test_rule(rng, step_posterior, args...; is_primitive=false)
        end

        # Run standard battery of LGSSM tests.
        test_interface(
            rng, model;
            rtol=1e-5,
            atol=1e-5,
            max_primal_allocs=25,
            max_forward_allocs=25,
            max_backward_allocs=25,
            check_allocs=TEST_ALLOC && storage.val isa SArrayStorage,
        )
    end
end
