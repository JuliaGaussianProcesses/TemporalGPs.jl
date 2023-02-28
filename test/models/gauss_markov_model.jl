using TemporalGPs: storage_type, is_of_storage_type

println("gauss_markov:")
@testset "gauss_markov" begin

    Dlats = [1, 7]
    Ns = [1, 11]
    tvs = [true, false]
    storages = [
        (name="Array{Float64}", val=ArrayStorage(Float64)),
        (name="SArray{Float64}", val=SArrayStorage(Float64)),
    ]

    # Dlatss = [7]
    # Ns = [11]
    # tvs = [true]

    @testset "time_varying=$tv, Dlat=$Dlat, N=$N, storage=$(storage.name)" for
        tv in tvs,
        Dlat in Dlats,
        N in Ns,
        storage in storages

        rng = MersenneTwister(123456)
        gmm = tv == true ?
            random_tv_gmm(rng, Forward(), Dlat, N, storage.val) :
            random_ti_gmm(rng, Forward(), Dlat, N, storage.val)

        @test eltype(gmm) == eltype(storage.val)
        @test storage_type(gmm) == storage.val

        @test length(gmm) == N
        @test getindex(gmm, N) isa TemporalGPs.SmallOutputLGC

        @test is_of_storage_type(gmm, storage.val)

        @testset "==" begin
            gmm_other = tv == true ?
                random_tv_gmm(rng, Forward(), Dlat, N, storage.val) :
                random_ti_gmm(rng, Forward(), Dlat, N, storage.val)
            @test gmm == gmm
            @test gmm != gmm_other
        end
    end
end
