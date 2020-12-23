using TemporalGPs: transition_dynamics, emission_dynamics, invert_dynamics

@testset "posterior_lgssm" begin
    rng = MersenneTwister(123456)

    Ns = [
        # 1,
        5,
    ]
    Dlats = [
        # 1,
        2,
    ]
    Dobss = [
        # 1,
        2,
    ]
    tvs = [
        # true,
        false,
    ]
    storages = [
        (name="dense storage Float64", val=ArrayStorage(Float64)),
        # (name="static storage Float64", val=SArrayStorage(Float64)),
        # (name="dense storage Float32", val=ArrayStorage(Float32)),
        # (name="static storage Float32", val=SArrayStorage(Float32)),
    ]

    @testset "(time_varying=$tv, Dlat=$Dlat, Dobs=$Dobs, $(storage.name)), N=$N" for
        tv in tvs,
        Dlat in Dlats,
        Dobs in Dobss,
        storage in storages,
        N in Ns

        # Print current iteration to prevent CI timing out.
        println("(time_varying=$tv, Dlat=$Dlat, Dobs=$Dobs, $(storage.name), $N)")

        # @testset "standardised AbstractSSM tests" begin
        #     model = tv ?
        #         random_tv_posterior_lgssm(rng, Dlat, Dobs, N, storage.val) :
        #         random_ti_posterior_lgssm(rng, Dlat, Dobs, N, storage.val)

        #     ssm_interface_tests(
        #         rng, model;
        #         rtol=1e-6, atol=1e-6, context=NoContext(),
        #     )
        # end
        @testset "constructing a PosteriorLGSSM" begin
            model = tv ?
                random_tv_lgssm(rng, Dlat, Dobs, N, storage.val) :
                random_ti_lgssm(rng, Dlat, Dobs, N, storage.val)
            y = rand(model)
            Σs = map(
                _ -> random_nice_psd_matrix(rng, dim_obs(model), storage.val),
                eachindex(y),
            )

            @testset "invert_dynamics" begin
                transitions = transition_dynamics(model)
                emissions = emission_dynamics(model)
                xfs = _filter(model, y)
                @inferred invert_dynamics(transitions[1], emissions[1], xfs[1], Σs[1])
                adjoint_test(
                    invert_dynamics, (transitions[1], emissions[1], xfs[1], Σs[1]);
                    context=NoContext(),
                )
            end
            @testset "posterior" begin
                @inferred posterior(model, y, Σs)
                @inferred posterior(model, y)
                adjoint_test(
                    posterior, (model, y, Σs);
                    context=NoContext(), check_infers=storage.val isa ArrayStorage,
                )
            end
        end
    end
end
