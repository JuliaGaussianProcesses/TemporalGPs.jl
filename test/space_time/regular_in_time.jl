using TemporalGPs: RegularInTime

@testset "regular_in_time" begin
    T = 11
    Nts = [rand(1:4) for _ in 1:T]
    xs = [randn(Nt) for Nt in Nts]
    ts = RegularSpacing(0.0, 0.3, T)
    x = RegularInTime(ts, xs)

    @test size(x) == length(collect(x))
end
