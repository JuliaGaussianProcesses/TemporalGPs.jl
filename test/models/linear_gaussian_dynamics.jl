@testset "linear_gaussian_dynamics" begin
    Dlats = [1, 3]
    Dobss = [1, 2]
    Dlats = [3]
    Dobss = [2]
    storages = [
        (name="dense storage Float64", val=ArrayStorage(Float64)),
        (name="static storage Float64", val=SArrayStorage(Float64)),
    ]

    @testset "(Dlat=$Dlat, Dobs=$Dobs, $(storage.name))" for
        Dlat in Dlats,
        Dobs in Dobss,
        storage in storages

        rng = MersenneTwister(123456)
        x = random_gaussian(rng, Dlat, storage.val)
        model = random_linear_gaussian_dynamics(rng, Dlat, Dobs, storage.val)
        y = rand(rng, TemporalGPs.predict(x, model))
        y = random_vector(rng, Dobs, storage.val)

        @testset "predict" begin
            adjoint_test(TemporalGPs.predict, (x, model))
            if storage.val isa SArrayStorage
                check_adjoint_allocations(TemporalGPs.predict, (x, model))
            end
        end
        @testset "correlate" begin
            adjoint_test(TemporalGPs.correlate, (x, model, y))
            if storage.val isa SArrayStorage
                check_adjoint_allocations(TemporalGPs.correlate, (x, model, y))
            end
        end
        @testset "decorrelate" begin
            adjoint_test(TemporalGPs.decorrelate, (x, model, y))
            if storage.val isa SArrayStorage
                check_adjoint_allocations(TemporalGPs.decorrelate, (x, model, y))
            end
        end
    end
end
