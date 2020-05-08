@testset "regular_data" begin
    t0 = randn()
    Δt = randn()
    N = 5
    x = RegularSpacing(t0, Δt, N)
    x_range = range(t0; step=Δt, length=N)

    @test size(x) == size(x_range)
    @test getindex(x, 3) ≈ getindex(x_range, 3)
    @test collect(x) ≈ collect(x_range)

    let
        x, back = Zygote.pullback(RegularSpacing, t0, Δt, N)

        Δ_t0 = randn()
        Δ_Δt = randn()
        @test back((t0 = Δ_t0, Δt = Δ_Δt, N=nothing)) == (Δ_t0, Δ_Δt, nothing)
    end
end
