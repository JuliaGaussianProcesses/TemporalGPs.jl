using Random
using TemporalGPs: RectilinearGrid, SpaceTimeGrid

function FiniteDifferences.to_vec(x::RectilinearGrid)
    v, tup_from_vec = to_vec((x.xl, x.xr))
    function RectilinearGrid_from_vec(v)
        tup = tup_from_vec(v)
        return RectilinearGrid(tup[1], tup[2])
    end
    return v, RectilinearGrid_from_vec
end

@testset "rectilinear_grid" begin
    rng = MersenneTwister(123456)
    Nl = 5
    Nr = 3
    xl = randn(rng, Nl)
    xr = randn(rng, Nr)

    X = RectilinearGrid(xl, xr)
    x = collect(X)

    @test x isa Vector
    @test eltype(x) == eltype(X)
    @test size(X) == (length(x),)
    @test length(x) == length(X)

    @test all(getindex.(Ref(x), 1:length(x)) .== x)
end
