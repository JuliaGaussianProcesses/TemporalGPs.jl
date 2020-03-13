using TemporalGPs: build_Σs

@testset "scalar" begin
    @testset "build_Σs" begin
        rng = MersenneTwister(123456)
        N = 11
        @testset "heteroscedastic" begin
            σ²_ns = exp.(randn(rng, N)) .+ 1e-3
            @test all(first.(build_Σs(σ²_ns)) == first.(σ²_ns))

            ΔΣs = SMatrix{1, 1}.(randn(rng, N))
            adjoint_test(build_Σs, ΔΣs, σ²_ns)
        end
        @testset "homoscedastic" begin
            σ²_n = exp(randn(rng)) + 1e-3
            σ²_ns = Fill(σ²_n, N)
            @test all(first.(build_Σs(σ²_ns)) == first.(σ²_ns))

            ΔΣs = (value=SMatrix{1, 1}(randn(rng)),)
            adjoint_test(σ²_n->build_Σs(Fill(σ²_n, N)), ΔΣs, σ²_n)
        end
    end
end
