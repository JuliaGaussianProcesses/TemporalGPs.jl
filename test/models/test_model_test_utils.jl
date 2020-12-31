@testset "model_test_utils" begin
    storages = [
        (name="dense storage", val=ArrayStorage(Float64)),
        (name="static storage", val=SArrayStorage(Float64)),
    ]
    @testset "storage = $(storage.name)" for storage in storages
        @testset "random_vector" begin
            rng = MersenneTwister(123456)
            a = random_vector(rng, 3, storage.val)
            @test is_of_storage_type(a, storage.val)
            @test length(a) == 3
        end
        @testset "random_matrix" begin
            rng = MersenneTwister(123456)
            A = random_matrix(rng, 4, 3, storage.val)
            @test is_of_storage_type(A, storage.val)
            @test size(A) == (4, 3)
        end
        @testset "random_nice_psd_matrix" begin
            rng = MersenneTwister(123456)
            Σ = random_nice_psd_matrix(rng, 11, storage.val)
            @test all(eigvals(Σ) .> 0)
            @test all(eigvals(Σ) .< 1)
            @test is_of_storage_type(Σ, storage.val)
        end
        @testset "random_gaussian" begin
            rng = MersenneTwister(123456)
            x = random_gaussian(rng, 3, storage.val)
            @test is_of_storage_type(x, storage.val)
            @test length(x.m) == 3
            @test size(x.P) == (3, 3)
            @test all(eigvals(x.P) .> 0) 
        end
    end
end
