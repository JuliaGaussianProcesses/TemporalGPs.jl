using FiniteDifferences
using Zygote

function FiniteDifferences.to_vec(x::RegularSpacing)
    function from_vec_RegularSpacing(x_vec)
        return RegularSpacing(x_vec[1], x_vec[2], x.N)
    end
    return [x.t0, x.Δt], from_vec_RegularSpacing
end

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

    let
        x, back = Zygote.pullback(RegularSpacing, t0, Δt, N)

        Δ_t0 = randn()
        Δ_Δt = randn()
        @test back((t0 = Δ_t0, Δt = Δ_Δt, N=nothing)) == (Δ_t0, Δ_Δt, nothing)

        test_rrule(RegularSpacing, randn(), rand(), 10; output_tangent=Tangent{RegularSpacing}(Δt=0.1, t0=0.2))
    end
end
