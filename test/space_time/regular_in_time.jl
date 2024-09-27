@testset "regular_in_time" begin
    T = 11
    Nts = [rand(1:4) for _ in 1:T]
    xs = [randn(Nt) for Nt in Nts]
    ts = RegularSpacing(0.0, 0.3, T)
    x = RegularInTime(ts, xs)

    @test prod(size(x)) == length(collect(x))

    @test all([getindex(x, n) for n in 1:length(x)] .== collect(x))
    @test_throws BoundsError x[0]
    @test_throws BoundsError x[-1]
    @test_throws BoundsError x[length(x) + 1]
end
