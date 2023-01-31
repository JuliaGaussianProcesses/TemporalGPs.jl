using FillArrays
using TemporalGPs

@testset "zygote_friendly_map" begin
    @testset "$name" for (name, f, x) in [
        ("Vector{Float64}", x -> sin(x) + cos(x) * exp(x), randn(100)),
        ("Fill{Float64}", x -> sin(x) + exp(x) + 5, Fill(randn(), 100)),
        ("Vector{Vector{Float64}}", sum, [randn(25) for _ in 1:33]),
        (
            "zip(Vector{Float64}, Fill{Float64})",
            x -> x[1] * x[2],
            zip(randn(5), Fill(1.0, 5)),
        ),
    ]
        @test TemporalGPs.zygote_friendly_map(f, x) ≈ map(f, x)
        # adjoint_test(x -> TemporalGPs.zygote_friendly_map(f, x), (x, ))
    end
end
