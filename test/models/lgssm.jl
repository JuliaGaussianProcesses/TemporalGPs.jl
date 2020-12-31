using TemporalGPs:
    predict,
    step_marginals,
    step_decorrelate,
    step_correlate,
    storage_type,
    is_of_storage_type

using Stheno: GP, GPC
using Zygote, StaticArrays

println("lgssm:")
@testset "lgssm" begin

    @testset "correctness" begin
        rng = MersenneTwister(123456)

        Ns = [1, 50]
        Dlats = [1, 2]
        Dobss = [1, 2]
        tvs = [true, false]
        # Ns = [100]
        # Dlats = [2]
        # Dobss = [2]
        # tvs = [true]
        storages = [
            (name="dense storage Float64", val=ArrayStorage(Float64)),
            (name="static storage Float64", val=SArrayStorage(Float64)),
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

            # Build LGSSM.
            model = tv ?
                random_tv_lgssm(rng, Forward(), Dlat, Dobs, N, storage.val) :
                random_ti_lgssm(rng, Forward(), Dlat, Dobs, N, storage.val)

            # Verify that model properties are as requested.
            @test eltype(model) == eltype(storage.val)
            @test storage_type(model) == storage.val

            @test length(model) == N
            @test getindex(model, N) isa TemporalGPs.ElementOfLGSSM

            @test is_of_storage_type(model, storage.val)

            y = first(rand(model))
            x0 = model.gmm.x0

            @testset "step_marginals" begin
                adjoint_test(step_marginals, (x0, model[1]))
                if storage.val isa SArrayStorage
                    check_adjoint_allocations(step_marginals, (x0, model[1]))
                end
            end
            @testset "step_decorrelate" begin
                adjoint_test(step_decorrelate, (x0, (model[1], y)))
                if storage.val isa SArrayStorage
                    check_adjoint_allocations(step_decorrelate, (x0, (model[1], y)))
                end
            end
            @testset "step_correlate" begin
                adjoint_test(step_correlate, (x0, (model[1], y)))
                if storage.val isa SArrayStorage
                    check_adjoint_allocations(step_correlate, (x0, (model[1], y)))
                end
            end

            # Run standard battery of LGSSM tests.
            ssm_interface_tests(
                rng, model;
                rtol=1e-5,
                atol=1e-5,
                context=NoContext(),
                max_primal_allocs=3,
                max_forward_allocs=10,
                max_backward_allocs=20,
                # check_allocs=false,
                check_allocs=storage.val isa SArrayStorage,
            )
        end
    end
end
