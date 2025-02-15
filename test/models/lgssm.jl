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
        (tv=:time_invariant, N=49, Dlat=3, Dobs=2, storage=storages.dense),
        (tv=:time_varying, N=49, Dlat=1, Dobs=1, storage=storages.dense),
        (tv=:time_varying, N=1, Dlat=3, Dobs=2, storage=storages.static),
        (tv=:time_invariant, N=49, Dlat=3, Dobs=2, storage=storages.static),
    ]
    orderings = [Forward(), Reverse()]
    Qs = [Val(:dense), Val(:diag)]

    @testset "($tv, $N, $Dlat, $Dobs, $(storage.name), $(emission.name), $order, $Q)" for (
            tv, N, Dlat, Dobs, storage
        ) in settings,
        emission in emission_types,
        order in orderings,
        Q in Qs

        # Print current iteration to prevent CI timing out.
        println(
            "(time_varying=$tv, N=$N, Dlat=$Dlat, Dobs=$Dobs, " *
            "storage=$(storage.name), emissions=$(emission.val), ordering=$order, " *
            "Q=$Q)",
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

        perf_flag = storage.val isa SArrayStorage ? :allocs : :none
        @testset "$f" for (f, args...) in Any[
            (step_marginals, x, model[1]),
            (step_logpdf, ordering(model[1]), x, (model[1], y)),
            (step_filter, ordering(model[1]), x, (model[1], y)),
            (invert_dynamics, x, x, model[1].transition),
            (step_posterior, ordering(model[1]), x, (model[1], y)),
        ]
            @test_opt target_modules = [TemporalGPs] f(args...)
            test_rule(rng, f, args...; is_primitive=false, interface_only=true, perf_flag)
        end

        # Run standard battery of LGSSM tests.
        test_interface(rng, model; check_allocs=false)
    end
end
