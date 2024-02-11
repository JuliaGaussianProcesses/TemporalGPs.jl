@testset "regular_data" begin
    t0 = randn()
    Δt = randn()
    N = 5
    x = RegularSpacing(t0, Δt, N)
    x_range = range(t0; step=Δt, length=N)

    @test size(x) == size(x_range)
    @test getindex(x, 3) ≈ getindex(x_range, 3)
    @test collect(x) ≈ collect(x_range)
    @test step(x) == step(x_range)
    @test length(x) == length(x_range)
end
