using TemporalGPs: are_harmonised

function test_harmonise(a, b; recurse=true)
    h = harmonise(a, b)
    @test h isa Tuple
    @test length(h) == 2
    @test are_harmonised(h[1], h[2])

    recurse && test_harmonise(b, a; recurse=false)
    h′ = harmonise(b, a)
    @test h isa Tuple
    @test length(h) == 2
    @test are_harmonised(h′[1], h′[2])
    @test are_harmonised(h[1], h′[1])
    @test are_harmonised(h[1], h′[2])
end

@testset "harmonise" begin
    test_harmonise(5.0, 4.0)

    @testset "AbstractZero" begin
        test_harmonise(5.0, Zero())
        test_harmonise(Zero(), randn(10))
        test_harmonise(Zero(), Zero())
    end

    @testset "Array" begin
        test_harmonise(randn(5), randn(5))
        test_harmonise(
            [(randn(), randn()) for _ in 1:10],
            [Composite{Any}(randn(), rand()) for _ in 1:10],
        )
    end

    @testset "Tuple / Composite{Tuple}" begin
        test_harmonise((5, 4), (5, 4))
        test_harmonise(Composite{Tuple}(5, 4), (5, 4))
        test_harmonise(Composite{Tuple}(5, 4), Composite{Tuple}(5, 4))

        test_harmonise((5, Composite{Tuple}(randn(5))), (5, (randn(5), )))
        test_harmonise(
            Composite{Any}(Composite{Any}(randn(5))),
            (Composite{Any}(randn(5)), ),
        )
    end

    @testset "NamedTuple / Composite{NamedTuple}" begin
        test_harmonise(Composite{Any}(; m=4, P=5), Composite{Gaussian}(; m=5, P=4))
        test_harmonise(Composite{Any}(; m=4, P=5), Composite{Any}(; m=4))
        test_harmonise(Composite{Any}(; m=5), Composite{Any}(; P=4))

        test_harmonise(Composite{Any}(; m=(5, 4)), Composite{Any}(; P=4))

        test_harmonise(Composite{Any}(; m=5, P=4), Gaussian(5, 4))
        test_harmonise(Composite{Any}(; P=4), Gaussian(4, 5))
    end
end
