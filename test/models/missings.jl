using TemporalGPs:
    x0,
    fill_in_missings,
    replace_observation_noise_cov,
    transform_model_and_obs

include("../test_util.jl")

println("missings:")
@testset "missings" begin

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
        (tv=:time_varying, N=5, Dlat=3, Dobs=2, storage=storages.dense),
        (tv=:time_varying, N=5, Dlat=3, Dobs=2, storage=storages.static),
        (tv=:time_invariant, N=5, Dlat=3, Dobs=2, storage=storages.dense),
        (tv=:time_invariant, N=5, Dlat=3, Dobs=2, storage=storages.static),
    ]
    orderings = [
        Forward(),
    ]

    # We can test this stuff by analytically dropping particular elements of the LGSSM
    # and computing the marginalised dynamics.
    @testset "($tv, $N, $Dlat, $Dobs, $(storage.name), $(emission.name), $order)" for
        (tv, N, Dlat, Dobs, storage) in settings,
        emission in emission_types,
        order in orderings

        # Print current iteration to prevent CI timing out.
        println(
            "(time_varying=$tv, N=$N, Dlat=$Dlat, Dobs=$Dobs, " *
            "storage=$(storage.name), emissions=$(emission.val), ordering=$order)",
        )

        # Build LGSSM.
        model = if emission.val ∈ (SmallOutputLGC, LargeOutputLGC)
            random_lgssm(
                rng, order, Val(tv), emission.val, Dlat, Dobs, N, Val(:dense), storage.val,
            )
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

        y = rand(model)
        x = TemporalGPs.x0(model)

        # Make some of the data missing.
        missings_idx = [3, 5, 9]
        missings_idx = [2, 4]
        presents_idx = setdiff(eachindex(y), missings_idx)
        y_missing = Vector{Union{eltype(y), Missing}}(undef, length(y))
        y_missing[missings_idx] .= missing
        y_missing[presents_idx] .= y[presents_idx]

        # Construct a modified LGSSM that is shorter than the original with analytically-
        # marginalised dynamics where there are missing data.
        As = model.transitions.As
        as = model.transitions.as
        Qs = model.transitions.Qs
        transitions_missing = map(eachindex(y_missing)) do n
            if n - 1 ∈ missings_idx
                A = As[n] * As[n-1]
                a = As[n] * as[n-1] + as[n]
                Q = As[n] * Qs[n-1] * As[n]' + Qs[n]
                return A, a, Q
            else
                return As[n], as[n], Qs[n]
            end
        end

        new_As = getindex.(transitions_missing, 1)[presents_idx]
        new_as = getindex.(transitions_missing, 2)[presents_idx]
        new_Qs = getindex.(transitions_missing, 3)[presents_idx]
        new_model = LGSSM(
            GaussMarkovModel(order, new_As, new_as, new_Qs, x0(model)),
            model.emissions[presents_idx],
        )
        new_y = y[presents_idx]

        @testset "logpdf" begin
            @test logpdf(new_model, new_y) ≈ logpdf(model, y_missing)
        end

        @testset "filter" begin
            marginal_filters = _filter(new_model, new_y)
            filters = _filter(model, y_missing)
            @test mean.(marginal_filters) ≈ mean.(filters)[presents_idx]
            @test cov.(marginal_filters) ≈ cov.(filters)[presents_idx]
        end

        @testset "posterior" begin
            new_posterior = posterior(new_model, new_y)
            post = posterior(model, y_missing)

            new_post_marginals = marginals(new_posterior)
            post_marginals = marginals(post)
            @test mean.(new_post_marginals) ≈ mean.(post_marginals)[presents_idx]
            @test cov.(new_post_marginals) ≈ cov.(post_marginals)[presents_idx]

            @test logpdf(new_posterior, new_y) ≈ logpdf(post, y_missing) rtol=1e-4
        end

        # Only test the bits of AD that we haven't tested before.
        @testset "AD: transform_model_and_obs" begin
            fdm = central_fdm(2, 1)
            adjoint_test(fill_in_missings, (model.emissions.Q, y_missing); fdm=fdm)
            adjoint_test(replace_observation_noise_cov, (model, model.emissions.Q))
            adjoint_test(transform_model_and_obs, (model, y_missing); fdm=fdm)
        end
    end

    storages = (
        dense=(name="dense storage Float64", val=ArrayStorage(Float64)),
    )
    emission_types = (
        small_output=(name="small output", val=SmallOutputLGC),
        large_output=(name="large output", val=LargeOutputLGC),
    )
    settings = [
        (tv=:time_varying, N=5, Dlat=3, Dobs=2, storage=storages.dense),
        (tv=:time_invariant, N=5, Dlat=3, Dobs=2, storage=storages.dense),
    ]
    orderings = [
        Forward(),
    ]

    @testset "missing-in-obs (tv=$(tv), N=$N, Dlat=$Dlat, emission=$emission" for
        (tv, N, Dlat, Dobs, storage) in settings,
        emission in emission_types,
        order in orderings

        println("missing-in-obs (tv=$(tv), N=$N, Dlat=$Dlat, emission=$emission")

        # Build LGSSM.
        model = random_lgssm(
            rng, order, Val(tv), emission.val, Dlat, Dobs, N, Val(:diag), storage.val,
        )

        # Verify the correct output types has been obtained.
        @test eltype(model.emissions) <: emission.val

        # Verify that model properties are as requested.
        @test storage_type(model) == storage.val

        @test length(model) == N
        @test getindex(model, N) isa TemporalGPs.ElementOfLGSSM

        # Generate some missing data.
        y = rand(model)
        y_missing = map(y) do yn
            yn_missing = Vector{Union{Missing, eltype(yn)}}(undef, length(yn))
            yn_missing .= yn
            yn_missing[randperm(length(yn))[1]] = missing
            return yn_missing
        end

        # Check logpdf and inference run, infer, and play nicely with AD.
        @inferred logpdf(model, y_missing)
        test_zygote_grad(y_missing) do y
            logpdf(model, y)
        end
        @inferred posterior(model, y_missing)
    end
end
