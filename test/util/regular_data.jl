function FiniteDifferences.to_vec(x::RegularSpacing)
    function from_vec_RegularSpacing(x_vec)
        return RegularSpacing(x[1], x[2], x.N)
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

        adjoint_test(
            (t0, Δt) -> RegularSpacing(t0, Δt, 5),
            (t0 = randn(), Δt = randn()),
            randn(), randn(),
        )
    end

    @testset "ExtendedRegularSpacing" begin
        @testset "Trivial Extension" begin
            x_trivial_ext = ExtendedRegularSpacing(x, 0, 0)
            @test size(x_trivial_ext) == size(x)
            @test step(x_trivial_ext) == step(x)
            @test x_trivial_ext ≈ x
            @test first(x_trivial_ext) == first(x)
            @test last(x_trivial_ext) == last(x)
            @test convert(RegularSpacing, x_trivial_ext) ≈ x_trivial_ext
        end

        @testset "Same Spacing" begin
            x_same_spacing = ExtendedRegularSpacing(x, 3, 4)
            @test step(x_same_spacing) == step(x)
            @test length(x_same_spacing) == length(x) + 3 + 4
            @test first(x_same_spacing) ≈ first(x) - 3 * step(x)
            @test last(x_same_spacing) ≈ last(x) + 4 * step(x)
            @test convert(RegularSpacing, x_same_spacing) ≈ x_same_spacing
        end

        @testset "Doubled Density" begin
            x_ext = ExtendedRegularSpacing(x, 2, 0, 0)
            @test step(x_ext) ≈ step(x) / 2
            @test length(x_ext) == 2 * length(x) - 1
            @test first(x_ext) ≈ first(x)
            @test last(x_ext) ≈ last(x)
            @test convert(RegularSpacing, x_ext) ≈ x_ext
        end

        @testset "Tripled density and extensions" begin
            x_ext = ExtendedRegularSpacing(x, 3, 4, 2)
            @test step(x_ext) ≈ step(x) / 3
            @test length(x_ext) == 3 * (length(x) - 1) + 1 + 4 + 2
            @test first(x_ext) ≈ first(x) - 4 * step(x_ext)
            @test last(x_ext) ≈ last(x) + 2 * step(x_ext)
            @test convert(RegularSpacing, x_ext) ≈ x_ext
        end
    end
end
