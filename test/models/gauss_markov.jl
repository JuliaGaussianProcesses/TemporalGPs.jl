println("gauss_markov:")
@testset "gauss_markov" begin

    Dlats = [1, 3, 5, 7]
    Dobss = [1, 2, 6, 7]
    Ns = [1, 3, 11]
    tvs = [true, false]
    storages = [
        (name="Array{Float64}", val=ArrayStorage(Float64)),
        (name="SArray{Float64}", val=SArrayStorage(Float64)),
    ]

    function test_name(tv, Dlat, Dobs, N, storage)
        return "GaussMarkovModel - " *
            "time_varying=$tv, Dlat=$Dlat, Dobs=$Dobs, N=$N, storage=$(storage.name)"
    end

    @testset "$(test_name(tv, Dlat, Dobs, N, storage))" for
        tv in tvs,
        Dlat in Dlats,
        Dobs in Dobss,
        N in Ns,
        storage in storages

        rng = MersenneTwister(123456)
        gmm = tv == true ?
            random_tv_gmm(rng, Dlat, Dobs, N, storage.val) :
            random_ti_gmm(rng, Dlat, Dobs, N, storage.val)

        @test eltype(gmm) == eltype(storage.val)
        @test TemporalGPs.storage_type(gmm) == storage.val

        @testset "==" begin
            gmm_other = tv == true ?
                random_tv_gmm(rng, Dlat, Dobs, N, storage.val) :
                random_ti_gmm(rng, Dlat, Dobs, N, storage.val)
            @test gmm == gmm
            @test gmm != gmm_other
        end

        @testset "mean / cov" begin
            m = mean(gmm)
            @test length(m) == N * Dobs

            P = cov(gmm)
            @test size(P) ==(N * Dobs, N * Dobs)
            @test all(eigvals(Symmetric(P)) .> -1e-9)
        end
    end
end
