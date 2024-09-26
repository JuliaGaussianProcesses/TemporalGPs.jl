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
